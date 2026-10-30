#!/usr/bin/env python3
"""
Decision-aligned re-analysis of the bottleneck pipeline. 
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import glob
import io
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

from bottleneck_sim.evaluation import HORIZONS, dm_test, holm, make_origins, run_rolling_evaluation  # noqa: E402
from bottleneck_sim.utils.features import PERIODS_PER_DAY as PPD  # noqa: E402

OUT = Path(__file__).with_name("decision_results.json")
from shared import HORIZON, N_PHASE, PHASE_SEED, WINDOW, evaluate, load_sim  # noqa: E402


# ---------------------------------------------------------------- episodic bottleneck
def episodic(sim, util, thr, fixed_origins):
    total_wait = sim["wait_Furnace"].fillna(0) * sim["jobs_Furnace"].fillna(0)
    ranked = np.sort(total_wait.to_numpy())[::-1]
    cong = util > thr
    weekday = util.index.dayofweek < 5
    by_hour = cong.groupby([np.where(weekday, "weekday", "weekend"), util.index.hour]).mean().unstack(0)
    return {
        "threshold": thr,
        "share_intervals_above_threshold": float(cong.mean()),
        "share_wait_above_threshold": float(total_wait[cong].sum() / total_wait.sum()),
        "share_intervals_above_0.9": float((util > 0.9).mean()),
        "share_wait_above_0.9": float(total_wait[util > 0.9].sum() / total_wait.sum()),
        "share_jobs_above_threshold": float(sim["jobs_Furnace"][cong].sum() / sim["jobs_Furnace"].sum()),
        "wait_per_job_congested_s": float(total_wait[cong].sum() / sim["jobs_Furnace"][cong].sum()),
        "wait_per_job_other_s": float(total_wait[~cong].sum() / sim["jobs_Furnace"][~cong].sum()),
        "share_wait_top1pct_intervals": float(ranked[: len(ranked) // 100].sum() / ranked.sum()),
        "share_wait_top5pct_intervals": float(ranked[: len(ranked) // 20].sum() / ranked.sum()),
        "util_mean": float(util.mean()), "util_median": float(util.median()),
        "share_intervals_idle": float((util < 0.01).mean()),
        "congestion_weekday": float(cong[weekday].mean()),
        "congestion_weekend": float(cong[~weekday].mean()),
        "congestion_sunday": float(cong[util.index.dayofweek == 6].mean()),
        "congestion_hours_0_to_4": float(cong[util.index.hour < 4].mean()),
        "congestion_weekday_08_to_21": float(cong[weekday & (util.index.hour >= 8) & (util.index.hour < 21)].mean()),
        "congestion_by_hour": {c: [float(v) for v in by_hour[c]] for c in by_hour.columns},
        "fixed_origin_start_days": sorted({util.index[p + 1].day_name() for p in fixed_origins}),
        "fixed_origin_start_hours": sorted({int(util.index[p + 1].hour) for p in fixed_origins}),
        "_wait_curve": ranked,
    }


def scenarios():
    rows = []
    for f in glob.glob(str(ROOT / "results" / "cache" / "sensitivity" / "*.json")):
        d = json.loads(Path(f).read_text())
        ms, wait = d["momentary_shares"], d["mean_wait_s"]
        rows.append({"id": d["id"], "kind": d["kind"], "margin": d["util_margin"],
                     "util_F": d["mean_util"]["Furnace"], "wait_F": wait["Furnace"],
                     "mom_F": ms["Furnace"], "runner_up": d["runner_up"],
                     "top_util": d["bottleneck"],
                     "top_wait": max(wait, key=wait.get), "top_mom": max(ms, key=ms.get),
                     "second_mom": sorted(ms, key=ms.get)[-2]})
    s = pd.DataFrame(rows)
    fit = stats.linregress(s["margin"], s["mom_F"])
    el = stats.linregress(np.log(s["util_F"]), np.log(s["wait_F"]))
    rho, p = stats.spearmanr(s["margin"], s["mom_F"])
    return {
        "n": len(s),
        "top_util_counts": s["top_util"].value_counts().to_dict(),
        "top_wait_counts": s["top_wait"].value_counts().to_dict(),
        "top_momentary_counts": s["top_mom"].value_counts().to_dict(),
        "runner_up_agreement": float((s["runner_up"] == s["second_mom"]).mean()),
        "furnace_share_below_half": int((s["mom_F"] < 0.5).sum()),
        "furnace_share_min": float(s["mom_F"].min()), "furnace_share_max": float(s["mom_F"].max()),
        "margin_min": float(s["margin"].min()), "margin_max": float(s["margin"].max()),
        "spearman_margin_share": float(rho), "spearman_p": float(p),
        "share_on_margin_slope": float(fit.slope), "share_on_margin_intercept": float(fit.intercept),
        "share_on_margin_r2": float(fit.rvalue ** 2),
        "margin_for_half_share": float((0.5 - fit.intercept) / fit.slope),
        "wait_util_elasticity": float(el.slope), "wait_util_elasticity_r2": float(el.rvalue ** 2),
        "wait_F_min": float(s["wait_F"].min()), "wait_F_max": float(s["wait_F"].max()),
        "util_F_min": float(s["util_F"].min()), "util_F_max": float(s["util_F"].max()),
        "_table": s,
    }


# ---------------------------------------------------------------- forecast diagnostics
def _event_scores(signal, event, days):
    tp = int((signal & event).sum()); fp = int((signal & ~event).sum())
    fn = int((~signal & event).sum()); tn = int((~signal & ~event).sum())
    pod = tp / (tp + fn) if tp + fn else float("nan")
    return {"recall": pod, "precision": tp / (tp + fp) if tp + fp else None,
            "pss": pod - fp / (fp + tn), "csi": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "false_alarms_per_day": fp / days}


def event_skill(A, FC, thr):
    """Point forecasts used as congestion warnings: at the fixed threshold, and at a
    cut-off matched to the number of congested periods (removes level shrinkage)."""
    out = {}
    for lab, H in HORIZONS.items():
        ev = (A[:, :H] > thr).ravel()
        entry = {"n_periods": int(ev.size), "n_congested": int(ev.sum()),
                 "share_congested": float(ev.mean())}
        if ev.sum() >= 5:
            days = ev.size / PPD
            for k, fc in FC.items():
                f = fc[:, :H].ravel()
                cut = float(np.sort(f)[::-1][int(ev.sum()) - 1])
                entry[k] = {"auc": float(roc_auc_score(ev, f)),
                            "fixed": _event_scores(f > thr, ev, days),
                            "rate_matched": {**_event_scores(f >= cut, ev, days), "cutoff": cut}}
        out[lab] = entry
    return out


def congestion_error(A, FC, thr, origins):
    """MAE split by the realised state. Conditioning on outcomes is NOT a proper
    evaluation (Lerch et al., 2017); reported only as a diagnostic of where errors fall."""
    out = {}
    for lab, H in HORIZONS.items():
        a = A[:, :H]; hi = a > thr
        entry = {"share_congested": float(hi.mean())}
        if hi.any():
            per = {k: np.array([np.abs(fc[i, :H] - a[i])[hi[i]].mean() if hi[i].any() else np.nan
                                for i in range(len(a))]) for k, fc in FC.items()}
            ok = ~np.isnan(per["ml_direct_xgb"])
            entry.update({
                "mae_congested": {k: float(np.abs(fc[:, :H] - a)[hi].mean()) for k, fc in FC.items()},
                "bias_congested": {k: float((fc[:, :H] - a)[hi].mean()) for k, fc in FC.items()},
                "mae_uncongested": {k: float(np.abs(fc[:, :H] - a)[~hi].mean()) for k, fc in FC.items()},
                "n_origins_with_congestion": int(ok.sum())})
            if ok.sum() >= 8:
                sub = np.asarray(origins)[ok]
                spacing = int(np.median(np.diff(sub))) if len(sub) > 1 else H
                lag = max(0, math.ceil(H / spacing) - 1)
                rows = [{"a": x, "b": y, **dm_test(per[x][ok], per[y][ok], lag)}
                        for x, y in itertools.combinations(FC, 2)]
                for r, p in zip(rows, holm([r["p"] for r in rows])):
                    r["p_holm"] = p
                entry["dm_congested"] = rows
        out[lab] = entry
    return out


def daily_peaks(A, FC, thr):
    n_days = A.shape[1] // PPD
    a = A.reshape(len(A), n_days, PPD)
    busy = a.max(2) > thr
    out = {"n_busy_days": int(busy.sum()), "actual_peak_mean": float(a.max(2)[busy].mean()),
           "actual_std": float(A.std(1).mean())}
    for k, fc in FC.items():
        f = fc.reshape(len(A), n_days, PPD)
        out[k] = {"forecast_peak_mean": float(f.max(2)[busy].mean()),
                  "peak_underestimate": float((a.max(2) - f.max(2))[busy].mean()),
                  "peak_timing_error_h": float((np.abs(f.argmax(2) - a.argmax(2)) * 0.25)[busy].mean()),
                  "forecast_std": float(fc.std(1).mean())}
    return out


def win_counts(A, FC):
    out = {}
    for lab, H in HORIZONS.items():
        mae = {k: np.abs(fc[:, :H] - A[:, :H]).mean(1) for k, fc in FC.items()}
        best = min(("seasonal_naive", "arima", "sarima"), key=lambda k: mae[k].mean())
        out[lab] = {"best_classical": best, "n": len(A),
                    "ml_beats_best_classical": int((mae["ml_direct_xgb"] < mae[best]).sum()),
                    "hybrid_beats_sarima": int((mae["hybrid_sarima_xgb"] < mae["sarima"]).sum())}
    return out


def forecast_block(ev, thr, origins):
    A, FC = ev["_test_actual"], ev["_test_fc"]
    return {"metrics": ev["metrics"], "dm": ev["dm"], "intervals": ev["intervals"],
            "event_skill": event_skill(A, FC, thr),
            "congestion_error": congestion_error(A, FC, thr, origins),
            "daily_peaks": daily_peaks(A, FC, thr), "win_counts": win_counts(A, FC)}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    sim = load_sim()
    util, usage = sim["util_Furnace"], sim["usage_kwh"]
    thr = float(util.loc[:"2018-04-30 23:45"].quantile(0.80))  # alert threshold, as in the pipeline

    fixed_o = make_origins(util.index, "2018-07-01", "2018-12-31", 7, WINDOW, HORIZON)
    first = util.index.get_indexer([pd.Timestamp("2018-07-01") - pd.Timedelta("15min")])[0]
    rng = np.random.default_rng(PHASE_SEED)
    phase_o = sorted(int(p) for p in rng.choice(np.arange(first, len(util) - HORIZON - 1), N_PHASE, replace=False))

    res = {"episodic": episodic(sim, util, thr, fixed_o), "scenarios": scenarios()}
    print("evaluating fixed weekly origins (cached)")
    res["fixed"] = forecast_block(evaluate(util, usage, fixed_o, args.jobs), thr, fixed_o)
    print(f"evaluating {N_PHASE} phase-randomized origins (fits any uncached forecasts)")
    ev = evaluate(util, usage, phase_o, args.jobs)
    res["phase"] = {"seed": PHASE_SEED, "origins": [str(util.index[p]) for p in phase_o],
                    "start_day_counts": pd.Series([util.index[p + 1].day_name() for p in phase_o]).value_counts().to_dict(),
                    "start_hour_counts_business": int(sum(8 <= util.index[p + 1].hour < 21 and util.index[p + 1].dayofweek < 5
                                                          for p in phase_o)),
                    **forecast_block(ev, thr, phase_o)}
    OUT.write_text(json.dumps(strip(res), indent=1, default=float))
    res["scenarios"]["_table"].to_csv(OUT.with_name("scenario_table.csv"), index=False)
    np.save(OUT.with_name("wait_curve.npy"), res["episodic"]["_wait_curve"])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
