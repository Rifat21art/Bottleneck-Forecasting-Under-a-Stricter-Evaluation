#!/usr/bin/env python3
"""
Robustness analyses requested in peer_review/full_review_report.md (items RR-1,
RR-5, RR-7 and R3-W3). Everything comes from the forecast cache; the only models
fitted here are the small XGBoost alert classifiers.

"""
import os
# Same thread settings as scripts/run_pipeline.py: XGBoost results depend on the thread count,
# and the published alert numbers were produced single-threaded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

try:
    import pmdarima  # noqa: F401
except ImportError:
    # Forecasts are read from results/cache/rolling only; a cache miss fails loudly.
    import types
    def _no_fit(*args, **kwargs):
        raise RuntimeError("pmdarima is not installed and a forecast was missing from the cache")
    sys.modules["pmdarima"] = types.SimpleNamespace(auto_arima=_no_fit)

import bottleneck_sim.evaluation as ev  # noqa: E402
import shared as da  # noqa: E402
from bottleneck_sim.forecasting.classifier import BottleneckAlertClassifier, alert_metrics, choose_cutoff  # noqa: E402

OUT = Path(__file__).with_name("robustness_results.json")
ALPHA = 0.10
CLASSICAL = ("seasonal_naive", "arima", "sarima")
PPD = 96


def origins_fixed(util):
    return ev.make_origins(util.index, "2018-07-01", "2018-12-31", 7, da.WINDOW, da.HORIZON)


def origins_phase(util):
    first = util.index.get_indexer([pd.Timestamp("2018-07-01") - pd.Timedelta("15min")])[0]
    rng = np.random.default_rng(da.PHASE_SEED)
    return sorted(int(p) for p in rng.choice(np.arange(first, len(util) - da.HORIZON - 1), da.N_PHASE, replace=False))


def per_origin_mae(A, F, H):
    return np.abs(F[:, :H] - A[:, :H]).mean(axis=1)


def dm_overlap(loss_a, loss_b, t, H):
    """DM statistic with window-overlap weights; t-distribution with n-1 df."""
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


def robust_tests(e, origins):
    A, FC, t = e["_test_actual"], e["_test_fc"], np.asarray(origins)
    out = {}
    for lab, H in ev.HORIZONS.items():
        mae = {k: per_origin_mae(A, f, H) for k, f in FC.items()}
        sel = thin(t, H)
        rows = []
        for a, b in itertools.combinations(FC, 2):
            rows.append({"a": a, "b": b, "overlap": dm_overlap(mae[a], mae[b], t, H),
                         "thinned": ev.dm_test(mae[a][sel], mae[b][sel], 0)})
        for key in ("overlap", "thinned"):
            for r, p in zip(rows, ev.holm([r[key]["p"] for r in rows])):
                r[key]["p_holm"] = p
        best = min(CLASSICAL, key=lambda k: mae[k].mean())
        key_row = next(r for r in rows if {r["a"], r["b"]} == {best, "ml_direct_xgb"})
        out[lab] = {"best_classical": best, "n": len(t), "n_thinned": int(sel.size),
                    "min_gap": int(np.diff(np.sort(t)).min()), "ml_vs_best": key_row, "all_pairs": rows}
    return out


def phase_strata(e, origins, util):
    A, FC = e["_test_actual"], e["_test_fc"]
    starts = util.index[np.asarray(origins) + 1]
    work = np.array([(s.dayofweek < 5) and (8 <= s.hour < 21) for s in starts])
    out = {}
    for name, mask in (("weekday_08_21", work), ("other", ~work)):
        t = np.asarray(origins)[mask]
        entry = {"n": int(mask.sum())}
        for lab, H in ev.HORIZONS.items():
            ml = per_origin_mae(A[mask], FC["ml_direct_xgb"][mask], H)
            sa = per_origin_mae(A[mask], FC["sarima"][mask], H)
            entry[lab] = {"mae_ml": float(ml.mean()), "mae_sarima": float(sa.mean()),
                          "gain_pct": float(100 * (1 - ml.mean() / sa.mean())),
                          "ml_wins": int((ml < sa).sum()),
                          "dm_overlap": dm_overlap(sa, ml, t, H)}
        out[name] = entry
    return out


def coverage_by_state(e, thr):
    A, FC, Q = e["_test_actual"], e["_test_fc"], e["interval_quantiles"]
    out = {}
    for k, f in FC.items():
        lo, hi = ev.interval_bounds(f, Q[k])
        inside = (A >= lo) & (A <= hi)
        score = (hi - lo) + (2 / ALPHA) * np.clip(lo - A, 0, None) + (2 / ALPHA) * np.clip(A - hi, 0, None)
        cong = A > thr
        out[k] = {}
        for lab, H in ev.HORIZONS.items():
            c, ins = cong[:, :H], inside[:, :H]
            out[k][lab] = {"coverage": float(ins.mean()),
                           "coverage_congested": float(ins[c].mean()) if c.any() else None,
                           "coverage_other": float(ins[~c].mean()),
                           "n_congested": int(c.sum()),
                           "interval_score": float(score[:, :H].mean())}
    return out


def alerts(util, usage, q, lead, budgets=(2.0,)):
    periods = (("2018-01-01", "2018-04-30 23:45"), ("2018-05-01", "2018-06-30 23:45"), ("2018-07-01", "2018-12-31 23:45"))
    clf = BottleneckAlertClassifier(threshold_quantile=q, lead=lead)
    clf.fit(util, usage, train_index=util.loc[periods[0][0]:periods[0][1]].index)
    cal = util.loc[periods[1][0]:periods[1][1]].index
    test = util.loc[periods[2][0]:periods[2][1]].index
    p_cal, y_cal = clf.predict_proba(cal), clf.true_labels(cal)
    y_test, p_test = clf.true_labels(test), clf.predict_proba(test)
    res = {"threshold": float(clf.threshold_), "positive_rate_test": float(y_test.mean())}
    for b in budgets:
        res[f"budget_{b:g}"] = alert_metrics(y_test, (p_test >= choose_cutoff(p_cal, y_cal, b)).astype(int))
    res["persistence"] = alert_metrics(y_test, clf.current_state(test))
    fa = alert_metrics(y_cal, clf.current_state(cal))["false_alarms_per_day"]
    res["matched_to_persistence"] = alert_metrics(y_test, (p_test >= choose_cutoff(p_cal, y_cal, fa)).astype(int))
    return res


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def main():
    sim = da.load_sim()
    util, usage = sim["util_Furnace"], sim["usage_kwh"]
    train = util.loc[:"2018-04-30 23:45"]
    test_half = util.loc["2018-07-01":"2018-12-31 23:45"]
    thr80 = float(train.quantile(0.80))
    designs = {"fixed": origins_fixed(util), "phase": origins_phase(util)}
    evals = {d: da.evaluate(util, usage, o, 1) for d, o in designs.items()}

    res = {"threshold_80": thr80}
    res["dm_robust"] = {d: robust_tests(evals[d], designs[d]) for d in designs}
    res["phase_strata"] = phase_strata(evals["phase"], designs["phase"], util)
    res["coverage"] = {d: coverage_by_state(evals[d], thr80) for d in designs}
    res["thresholds"] = {}
    for q in (0.70, 0.80, 0.90):
        thr = float(train.quantile(q))
        base = float((test_half > thr).mean())
        entry = {"threshold": thr, "base_rate": base}
        for d in designs:
            A = evals[d]["_test_actual"]
            entry[d] = {lab: {"share": float((A[:, :H] > thr).mean()), "ratio": float((A[:, :H] > thr).mean() / base)}
                        for lab, H in ev.HORIZONS.items()}
        entry["alerts_1h"] = alerts(util, usage, q, 4)
        entry["alerts_15min"] = alerts(util, usage, q, 1)
        res["thresholds"][f"q{int(q * 100)}"] = entry
    res["budgets_1h_q80"] = alerts(util, usage, 0.80, 4, budgets=(0.5, 1.0, 2.0, 4.0, 8.0))

    # consistency with the published files
    R = json.loads((ROOT / "results" / "results.json").read_text())
    res["checks"] = {
        "coverage_matches_results_json": all(
            abs(res["coverage"]["fixed"][k][h]["coverage"] - R["forecasting"]["intervals"][k][h]["coverage"]) < 1e-12
            for k in R["forecasting"]["intervals"] for h in ev.HORIZONS),
        "alerts_q80_match_results_json": abs(res["thresholds"]["q80"]["alerts_1h"]["budget_2"]["recall"]
                                             - R["alerts"]["1h"]["test"]["recall"]) < 1e-12,
    }
    OUT.write_text(json.dumps(strip(res), indent=1))

    print("checks:", res["checks"])
    for d in designs:
        for lab in ev.HORIZONS:
            r = res["dm_robust"][d][lab]
            m = r["ml_vs_best"]
            print(f"{d:5s} {lab:3s} vs {r['best_classical']:14s} overlap stat {m['overlap']['stat']:6.2f} p_holm {m['overlap']['p_holm']:.4g} "
                  f"| thinned n={r['n_thinned']:2d} stat {m['thinned']['stat']:6.2f} p_holm {m['thinned']['p_holm']:.4g}")
    for s, e in res["phase_strata"].items():
        print(s, e["n"], {lab: (round(e[lab]["gain_pct"], 1), e[lab]["ml_wins"], round(e[lab]["dm_overlap"]["p"], 4)) for lab in ev.HORIZONS})
    for d in designs:
        c = res["coverage"][d]["ml_direct_xgb"]
        print(d, "XGB cov cong/other:", {lab: (None if c[lab]["coverage_congested"] is None else round(100 * c[lab]["coverage_congested"], 1),
                                              round(100 * c[lab]["coverage_other"], 1), round(c[lab]["interval_score"], 3)) for lab in ev.HORIZONS})
    for q, e in res["thresholds"].items():
        print(q, round(e["threshold"], 3), "base", round(e["base_rate"], 4),
              "fixed R", [round(e["fixed"][lab]["ratio"], 2) for lab in ev.HORIZONS],
              "phase R", [round(e["phase"][lab]["ratio"], 2) for lab in ev.HORIZONS],
              "1h recall budget/persist/matched", round(e["alerts_1h"]["budget_2"]["recall"], 3),
              round(e["alerts_1h"]["persistence"]["recall"], 3), round(e["alerts_1h"]["matched_to_persistence"]["recall"], 3))
    b = res["budgets_1h_q80"]
    print("budgets:", {k: (round(v["precision"], 3), round(v["recall"], 3), round(v["false_alarms_per_day"], 2)) for k, v in b.items() if k.startswith("budget")})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
