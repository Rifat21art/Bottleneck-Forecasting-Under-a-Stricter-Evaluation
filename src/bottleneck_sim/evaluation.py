"""
Rolling-origin, multi-horizon evaluation of every registered forecaster.

"""
import hashlib
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .forecasting import MODEL_REGISTRY

EVAL_VERSION = 1
HORIZONS = {"1h": 4, "4h": 16, "1d": 96, "10d": 960}
INTERVAL_BUCKETS = [(1, 4), (5, 16), (17, 96), (97, 960)]  # inclusive, 1-based steps
SEASON = 96

_G = {}


# ---------------------------------------------------------------- origins
def make_origins(index, first_test_day, last_test_day, every_days, window, horizon):
    """Origins = the timestamp just before each test-start day (00:00), kept if the
    training window, one extra week of history, and the full horizon fit in `index`."""
    starts = pd.date_range(first_test_day, last_test_day, freq=f"{every_days}D")
    positions = []
    for day in starts:
        pos = index.get_indexer([day - pd.Timedelta("15min")])[0]
        if pos >= window + 7 * SEASON and pos + horizon < len(index):
            positions.append(int(pos))
    return positions


# ---------------------------------------------------------------- workers
def _init_worker(y, usage, cache_dir):
    _G.update(y=y, usage=usage, cache_dir=Path(cache_dir))
    _G["data_key"] = hashlib.sha1(
        np.ascontiguousarray(np.r_[y.to_numpy(float), usage.to_numpy(float)]).tobytes()).hexdigest()[:12]


def _cache_path(key, params_json, origin_ts, window, horizon):
    h = hashlib.sha1(json.dumps([EVAL_VERSION, _G["data_key"], key, params_json, str(origin_ts),
                                 window, horizon]).encode()).hexdigest()[:20]
    return _G["cache_dir"] / f"{key}_{h}.npy"


def _run_origin(pos, specs, window, horizon):
    y, usage = _G["y"], _G["usage"]
    train = y.iloc[pos - window + 1: pos + 1]
    idx = y.index[pos + 1: pos + 1 + horizon]
    usage_train = usage.iloc[: pos + 1]
    fit_cache, out = {}, {}
    for key, params_json in specs:
        path = _cache_path(key, params_json, y.index[pos], window, horizon)
        if path.exists():
            out[(key, params_json)] = np.load(path)
            continue
        model = MODEL_REGISTRY[key](**json.loads(params_json))
        model.fit(train, usage_series=usage_train, fit_cache=fit_cache)
        fc = model.predict(horizon, index=idx).reindex(idx).to_numpy(dtype=float)
        np.save(path, fc)
        out[(key, params_json)] = fc
    return pos, out


def _execute(y, usage, cache_dir, specs_by_origin, window, horizon, jobs, label):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # SARIMA-style base models before models that reuse their fits
    order = {k: (1 if MODEL_REGISTRY[k].depends_on else 0) for k in MODEL_REGISTRY}
    specs_by_origin = {p: sorted(s, key=lambda ks: order[ks[0]]) for p, s in specs_by_origin.items()}
    results = {}
    n = len(specs_by_origin)
    if jobs <= 1:
        _init_worker(y, usage, cache_dir)
        for i, (pos, specs) in enumerate(specs_by_origin.items(), 1):
            results[pos] = _run_origin(pos, specs, window, horizon)[1]
            print(f"      {label} origin {i}/{n} done", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker,
                             initargs=(y, usage, str(cache_dir))) as pool:
        futs = [pool.submit(_run_origin, pos, specs, window, horizon) for pos, specs in specs_by_origin.items()]
        for i, fut in enumerate(as_completed(futs), 1):
            pos, out = fut.result()
            results[pos] = out
            print(f"      {label} origin {i}/{n} done ({y.index[pos + 1].date()})", flush=True)
    return results


# ---------------------------------------------------------------- metrics
def mase_scale(train, m=SEASON):
    train = np.asarray(train, dtype=float)
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def per_origin_errors(actual, forecast, scales):
    """dict horizon -> dict(MAE, MSE, MASE) arrays of shape (n_origins,)"""
    err = forecast - actual
    out = {}
    for label, H in HORIZONS.items():
        mae = np.abs(err[:, :H]).mean(axis=1)
        out[label] = {"MAE": mae, "MSE": (err[:, :H] ** 2).mean(axis=1), "MASE": mae / scales}
    return out


def summarize_errors(errs):
    return {label: {"MAE": float(e["MAE"].mean()), "RMSE": float(np.sqrt(e["MSE"].mean())),
                    "MASE": float(e["MASE"].mean()), "n_origins": int(e["MAE"].size)}
            for label, e in errs.items()}


def dm_test(loss_a, loss_b, lag=0):
    """Diebold-Mariano test with HLN correction. Negative statistic => A has lower loss."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        var = gamma[0]
    if var <= 0:
        return {"mean_diff": float(d.mean()), "stat": 0.0, "p": 1.0}
    h = lag + 1
    stat = d.mean() / math.sqrt(var / n)
    stat *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(p)}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def interval_quantiles(val_errors, alpha=0.10):
    """Per-bucket (lo, hi) quantiles of validation errors e = forecast - actual (n_origins x horizon)."""
    q = []
    for a, b in INTERVAL_BUCKETS:
        e = np.sort(val_errors[:, a - 1:b].ravel())
        n = e.size
        k_hi = min(n - 1, math.ceil((n + 1) * (1 - alpha / 2)) - 1)
        k_lo = max(0, math.floor((n + 1) * (alpha / 2)) - 1)
        q.append((float(e[k_lo]), float(e[k_hi])))
    return q


def interval_bounds(forecast, quantiles):
    """Interval for the actual value: the quantiles describe e = forecast - actual, so the
    upper error quantile sets the lower bound and vice versa."""
    lo, hi = np.empty_like(forecast), np.empty_like(forecast)
    for (a, b), (ql, qh) in zip(INTERVAL_BUCKETS, quantiles):
        lo[..., a - 1:b] = forecast[..., a - 1:b] - qh
        hi[..., a - 1:b] = forecast[..., a - 1:b] - ql
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def interval_performance(actual, forecast, quantiles):
    lo, hi = interval_bounds(forecast, quantiles)
    inside = (actual >= lo) & (actual <= hi)
    return {label: {"coverage": float(inside[:, :H].mean()), "mean_width": float((hi - lo)[:, :H].mean())}
            for label, H in HORIZONS.items()}


# ---------------------------------------------------------------- orchestration
def _grid(key, grid_limit):
    grid = MODEL_REGISTRY[key].param_grid or [{}]
    return grid[:grid_limit] if grid_limit else grid


def _stack(results, origins, spec):
    return np.vstack([results[pos][spec] for pos in origins])


def run_rolling_evaluation(y, usage, val_origins, test_origins, window, horizon, cache_dir,
                           jobs=1, grid_limit=None, alpha=0.10):
    models = list(MODEL_REGISTRY)
    actual = lambda origins: np.vstack([y.to_numpy(float)[p + 1: p + 1 + horizon] for p in origins])
    scales = lambda origins: np.array([mase_scale(y.to_numpy(float)[p - window + 1: p + 1]) for p in origins])
    val_actual, val_scales = actual(val_origins), scales(val_origins)

    def score(fc):
        errs = per_origin_errors(val_actual, fc, val_scales)
        return float(np.mean([e["MASE"].mean() for e in errs.values()]))

    tuned, val_scores, val_fc = {}, {}, {}
    independent = [k for k in models if not MODEL_REGISTRY[k].depends_on]
    dependent = [k for k in models if MODEL_REGISTRY[k].depends_on]
    for phase, keys in (("tuning", independent), ("tuning (dependent models)", dependent)):
        if not keys:
            continue
        specs = []
        for k in keys:
            inject = {arg: tuned[dep] for arg, dep in MODEL_REGISTRY[k].depends_on.items()}
            specs += [(k, json.dumps({**params, **inject}, sort_keys=True)) for params in _grid(k, grid_limit)]
        print(f"    {phase}: {len(specs)} configurations x {len(val_origins)} validation origins")
        res = _execute(y, usage, cache_dir, {p: specs for p in val_origins}, window, horizon, jobs, "validation")
        for k in keys:
            cands = [s for s in specs if s[0] == k]
            scored = {s[1]: score(_stack(res, val_origins, s)) for s in cands}
            best = min(scored, key=scored.get)
            tuned[k] = json.loads(best)
            val_scores[k] = [{"params": json.loads(pj), "mean_MASE": v} for pj, v in scored.items()]
            val_fc[k] = _stack(res, val_origins, (k, best))
            print(f"      {k}: best validation mean MASE {scored[best]:.3f} with {best}")

    specs = [(k, json.dumps(tuned[k], sort_keys=True)) for k in models]
    print(f"    test: {len(models)} tuned models x {len(test_origins)} test origins")
    res = _execute(y, usage, cache_dir, {p: specs for p in test_origins}, window, horizon, jobs, "test")
    test_actual, test_scales = actual(test_origins), scales(test_origins)
    test_fc = {k: _stack(res, test_origins, (k, pj)) for k, pj in specs}

    errs = {k: per_origin_errors(test_actual, fc, test_scales) for k, fc in test_fc.items()}
    metrics = {label: {k: summarize_errors(errs[k])[label] for k in models} for label in HORIZONS}

    spacing = int(np.median(np.diff(test_origins))) if len(test_origins) > 1 else horizon
    dm = {}
    for label, H in HORIZONS.items():
        lag = max(0, math.ceil(H / spacing) - 1)
        rows = [{"a": a, "b": b, **dm_test(errs[a][label]["MAE"], errs[b][label]["MAE"], lag)}
                for a, b in itertools.combinations(models, 2)]
        for r, p_adj in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = p_adj
        dm[label] = rows

    quantiles = {k: interval_quantiles(val_fc[k] - val_actual, alpha) for k in models}
    intervals = {k: interval_performance(test_actual, test_fc[k], quantiles[k]) for k in models}

    return {
        "models": {k: MODEL_REGISTRY[k].name for k in models},
        "tuned_params": tuned, "validation_scores": val_scores,
        "val_origins": [str(y.index[p]) for p in val_origins],
        "test_origins": [str(y.index[p]) for p in test_origins],
        "window": window, "horizon": horizon, "alpha": alpha,
        "metrics": metrics, "dm": dm,
        "interval_quantiles": quantiles, "intervals": intervals,
        # arrays for figures (not serialized)
        "_test_actual": test_actual, "_test_fc": test_fc, "_test_origin_pos": test_origins,
    }
