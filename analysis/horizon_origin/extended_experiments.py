
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import bottleneck_sim.evaluation as ev  # noqa: E402
import shared as da  # noqa: E402
import sarima_energy  # noqa: E402,F401  (registers "sarima_energy")

OUT = HERE / "extended_results.json"
ORIGINAL = ["seasonal_naive", "arima", "sarima", "hybrid_sarima_xgb", "ml_direct_xgb"]
CLASSICAL = ("seasonal_naive", "arima", "sarima")
ALPHA = 0.10


# ---------------------------------------------------------------- origin designs
def origins_random(util, seed):
    first = util.index.get_indexer([pd.Timestamp("2018-07-01") - pd.Timedelta("15min")])[0]
    rng = np.random.default_rng(seed)
    return sorted(int(p) for p in rng.choice(np.arange(first, len(util) - da.HORIZON - 1), da.N_PHASE, replace=False))


def origins_shifted(util, first_start="2018-07-04 10:00", n=25):
    starts = pd.date_range(first_start, periods=n, freq="7D")
    pos = [int(util.index.get_loc(s)) - 1 for s in starts]
    assert all(p + da.HORIZON < len(util) for p in pos)
    return pos


# ---------------------------------------------------------------- tests
def per_origin_mae(A, F, H):
    return np.abs(F[:, :H] - A[:, :H]).mean(axis=1)


def dm_overlap(loss_a, loss_b, t, H):
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    dc = d - d.mean()
    w = np.clip(1 - np.abs(np.subtract.outer(t, t)) / H, 0, None)
    var = float(dc @ w @ dc) / n ** 2
    stat = d.mean() / math.sqrt(var) if var > 0 else 0.0
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), n - 1))}


def thin(t, H):
    keep, last = [], -np.inf
    for i, ti in enumerate(t):
        if ti - last >= H:
            keep.append(i)
            last = ti
    return np.array(keep)


def holm_into(rows, key):
    for r, p in zip(rows, ev.holm([r[key]["p"] for r in rows])):
        r[key]["p_holm"] = p


def analyse(e, origins, util, thr, base):
    A, FC, Q = e["_test_actual"], e["_test_fc"], e["interval_quantiles"]
    t = np.asarray(origins)
    gap = int(np.median(np.diff(t))) if len(t) > 1 else da.HORIZON
    starts = util.index[t + 1]
    out = {"n": len(t), "median_gap": gap, "min_gap": int(np.diff(t).min()),
           "n_start_weekday_08_21": int(sum((s.dayofweek < 5) and (8 <= s.hour < 21) for s in starts)),
           "metrics": e["metrics"], "tuned_params": e["tuned_params"],
           "validation_scores": e["validation_scores"], "horizons": {}}
    for lab, H in ev.HORIZONS.items():
        mae = {k: per_origin_mae(A, f, H) for k, f in FC.items()}
        lag = max(0, math.ceil(H / gap) - 1)
        sel = thin(t, H)
        pairs = []
        for a, b in itertools.combinations(ORIGINAL, 2):
            pairs.append({"a": a, "b": b, "hln": ev.dm_test(mae[a], mae[b], lag), "overlap": dm_overlap(mae[a], mae[b], t, H),
                          "thinned": ev.dm_test(mae[a][sel], mae[b][sel], 0)})
        for key in ("hln", "overlap", "thinned"):
            holm_into(pairs, key)
        best = min(CLASSICAL, key=lambda k: mae[k].mean())
        key_pair = next(r for r in pairs if {r["a"], r["b"]} == {best, "ml_direct_xgb"})
        abl = []
        for a, b in (("sarima", "sarima_energy"), ("sarima_energy", "ml_direct_xgb"), ("sarima_energy", "hybrid_sarima_xgb")):
            abl.append({"a": a, "b": b, "gain_pct": float(100 * (1 - mae[b].mean() / mae[a].mean())),
                        "b_wins": int((mae[b] < mae[a]).sum()), "hln": ev.dm_test(mae[a], mae[b], lag),
                        "overlap": dm_overlap(mae[a], mae[b], t, H)})
        for key in ("hln", "overlap"):
            holm_into(abl, key)
        cong = A[:, :H] > thr
        cov = {}
        for k, f in FC.items():
            lo, hi = ev.interval_bounds(f, Q[k])
            inside = ((A >= lo) & (A <= hi))[:, :H]
            score = ((hi - lo) + (2 / ALPHA) * np.clip(lo - A, 0, None) + (2 / ALPHA) * np.clip(A - hi, 0, None))[:, :H]
            cov[k] = {"coverage": float(inside.mean()),
                      "coverage_congested": float(inside[cong].mean()) if cong.any() else None,
                      "coverage_other": float(inside[~cong].mean()), "interval_score": float(score.mean())}
        out["horizons"][lab] = {
            "lag": lag, "n_thinned": int(sel.size), "best_classical": best,
            "ml_gain_pct": float(100 * (1 - mae["ml_direct_xgb"].mean() / mae[best].mean())),
            "ml_wins": int((mae["ml_direct_xgb"] < mae[best]).sum()),
            "ml_vs_best": key_pair, "all_pairs": pairs, "ablation": abl,
            "congested": int(cong.sum()), "scored": int(cong.size), "ratio": float(cong.mean() / base),
            "coverage": cov}
    return out


def strata(ev_by_seed, origins_by_seed, util):
    """Pool the randomized designs (unique origins) and compare XGBoost with SARIMA by start phase."""
    rows = {}
    for seed, e in ev_by_seed.items():
        for i, p in enumerate(origins_by_seed[seed]):
            rows.setdefault(p, (e["_test_actual"][i], {k: e["_test_fc"][k][i] for k in ("sarima", "ml_direct_xgb", "sarima_energy")}))
    pos = np.array(sorted(rows))
    A = np.vstack([rows[p][0] for p in pos])
    F = {k: np.vstack([rows[p][1][k] for p in pos]) for k in ("sarima", "ml_direct_xgb", "sarima_energy")}
    starts = util.index[pos + 1]
    work = np.array([(s.dayofweek < 5) and (8 <= s.hour < 21) for s in starts])
    out = {"n_unique": int(pos.size)}
    for name, m in (("weekday_08_21", work), ("other", ~work)):
        g = {"n": int(m.sum())}
        for lab, H in ev.HORIZONS.items():
            ml, sa, se = (per_origin_mae(A[m], F[k][m], H) for k in ("ml_direct_xgb", "sarima", "sarima_energy"))
            g[lab] = {"gain_ml_vs_sarima": float(100 * (1 - ml.mean() / sa.mean())), "ml_wins": int((ml < sa).sum()),
                      "p_overlap": dm_overlap(sa, ml, pos[m], H)["p"],
                      "gain_energy_vs_sarima": float(100 * (1 - se.mean() / sa.mean())),
                      "gain_ml_vs_energy": float(100 * (1 - ml.mean() / se.mean()))}
        out[name] = g
    return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    sim = da.load_sim()
    util, usage = sim["util_Furnace"], sim["usage_kwh"]
    thr = float(util.loc[:"2018-04-30 23:45"].quantile(0.80))
    base = float((util.loc["2018-07-01":"2018-12-31 23:45"] > thr).mean())
    val = ev.make_origins(util.index, "2018-03-01", "2018-06-20", 14, da.WINDOW, da.HORIZON)
    designs = {"fixed": ev.make_origins(util.index, "2018-07-01", "2018-12-31", 7, da.WINDOW, da.HORIZON),
               "phase7": origins_random(util, 7), "shifted": origins_shifted(util),
               "phase11": origins_random(util, 11), "phase23": origins_random(util, 23)}
    evals, res = {}, {"threshold": thr, "base_rate": base, "designs": {}}
    for name, o in designs.items():
        print(f"== {name}: {len(o)} origins", flush=True)
        evals[name] = ev.run_rolling_evaluation(util, usage, val, o, da.WINDOW, da.HORIZON,
                                                ROOT / "results" / "cache" / "rolling", jobs=args.jobs)
        res["designs"][name] = analyse(evals[name], o, util, thr, base)
        res["designs"][name]["first_start"] = str(util.index[o[0] + 1])
    res["strata_pooled"] = strata({s: evals[f"phase{s}"] for s in (7, 11, 23)},
                                  {s: designs[f"phase{s}"] for s in (7, 11, 23)}, util)
    # consistency with the published five-model results
    R = json.loads((ROOT / "results" / "results.json").read_text())
    Dp = json.loads((ROOT / "analysis" / "decision" / "decision_results.json").read_text())
    res["checks"] = {
        "fixed_matches_results_json": all(abs(res["designs"]["fixed"]["metrics"][h][k]["MAE"] - R["forecasting"]["metrics"][h][k]["MAE"]) < 1e-12
                                          for h in ev.HORIZONS for k in ORIGINAL),
        "phase7_matches_decision_results": all(abs(res["designs"]["phase7"]["metrics"][h][k]["MAE"] - Dp["phase"]["metrics"][h][k]["MAE"]) < 1e-12
                                               for h in ev.HORIZONS for k in ORIGINAL),
    }
    OUT.write_text(json.dumps(strip(res), indent=1))
    print("checks:", res["checks"])
    for name, d in res["designs"].items():
        print(f"\n{name}: n={d['n']} work-hour starts={d['n_start_weekday_08_21']} first={d['first_start']}")
        for lab in ev.HORIZONS:
            h = d["horizons"][lab]
            k = h["ml_vs_best"]
            ab = {f"{x['a']}>{x['b']}": (round(x["gain_pct"], 1), round(x["overlap"]["p_holm"], 4)) for x in h["ablation"]}
            print(f"  {lab:3s} R={h['ratio']:.2f} XGB vs {h['best_classical']:14s} gain {h['ml_gain_pct']:5.1f}% wins {h['ml_wins']:2d} "
                  f"p_holm hln {k['hln']['p_holm']:.3g} ovl {k['overlap']['p_holm']:.3g} thin {k['thinned']['p_holm']:.3g} | ablation {ab}")
    print("\nstrata pooled:", json.dumps(res["strata_pooled"], indent=0)[:1500])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
