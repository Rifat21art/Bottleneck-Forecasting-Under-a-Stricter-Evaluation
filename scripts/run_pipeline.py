#!/usr/bin/env python3
"""
End-to-end pipeline:
  1. load real data                       4. rolling-origin tuning + test of every forecaster
  2. DES simulation + bottleneck ID       5. bottleneck alerts under a false-alarm budget
  3. parameter-sensitivity analysis       6. write results/results.json + all figures

Usage:
    python scripts/run_pipeline.py                    # full run (~40 min on 4 threads, resumable)
    python scripts/run_pipeline.py --quick            # smoke test (~2 min)
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per worker process

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from statsmodels.stats.diagnostic import acorr_ljungbox

import bottleneck_sim
from bottleneck_sim.data_loader import load_steel_data
from bottleneck_sim.simulation import (run_simulation, identify_bottleneck, station_summary,
                                       DEFAULT_STATIONS, SIM_VERSION)
from bottleneck_sim.sensitivity import run_sensitivity, summarize_sensitivity
from bottleneck_sim.evaluation import make_origins, run_rolling_evaluation, HORIZONS
from bottleneck_sim.forecasting import MODEL_REGISTRY
from bottleneck_sim.forecasting.classifier import BottleneckAlertClassifier, choose_cutoff, alert_metrics
from bottleneck_sim.utils.features import PERIODS_PER_DAY
from bottleneck_sim.figures import make_all_figures

ROOT = Path(__file__).resolve().parents[1]
ALERT_LEADS = {"15min": 1, "1h": 4}


def load_or_run_simulation(usage, cache_path, force):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        try:
            sim = pd.read_pickle(cache_path)
            if sim.attrs.get("sim_version") == SIM_VERSION:
                print(f"[2/6] Loading cached simulation output from {cache_path}")
                return sim
            print(f"[2/6] Cached simulation is from an older simulator version; re-running")
        except Exception as exc:  # e.g. pickle written by an incompatible pandas version
            print(f"[2/6] Could not read simulation cache ({type(exc).__name__}); re-running")
    else:
        print("[2/6] Running discrete-event simulation (this can take ~1 min)...")
    sim = run_simulation(usage)
    sim.to_pickle(cache_path)
    return sim


def evaluate_alerts(util, usage, periods, lead, max_fa_per_day):
    train, calib, test = periods
    clf = BottleneckAlertClassifier(threshold_quantile=0.80, lead=lead)
    clf.fit(util, usage, train_index=util.loc[train[0]:train[1]].index)
    calib_idx = util.loc[calib[0]:calib[1]].index
    test_idx = util.loc[test[0]:test[1]].index
    clf.cutoff_ = choose_cutoff(clf.predict_proba(calib_idx), clf.true_labels(calib_idx), max_fa_per_day)

    y_test = clf.true_labels(test_idx)
    proba = clf.predict_proba(test_idx)
    model = alert_metrics(y_test, (proba >= clf.cutoff_).astype(int))
    calib_m = alert_metrics(clf.true_labels(calib_idx), clf.predict(calib_idx))
    persistence = alert_metrics(y_test, clf.current_state(test_idx))
    # like-for-like comparison: classifier cut-off allowed the persistence rule's
    # false-alarm rate, both measured on the calibration period
    persistence_calib_fa = alert_metrics(clf.true_labels(calib_idx), clf.current_state(calib_idx))["false_alarms_per_day"]
    matched_cut = choose_cutoff(clf.predict_proba(calib_idx), clf.true_labels(calib_idx), persistence_calib_fa)
    matched = alert_metrics(y_test, (proba >= matched_cut).astype(int))
    prec, rec, _ = precision_recall_curve(y_test, proba)
    return {
        "lead_periods": lead, "threshold": float(clf.threshold_), "cutoff": clf.cutoff_,
        "positive_rate_test": float(y_test.mean()), "test_days": len(y_test) / PERIODS_PER_DAY,
        "calibration": calib_m, "test": model, "persistence_baseline": persistence,
        "matched_to_persistence": {**matched, "cutoff": matched_cut, "calibration_fa_budget": persistence_calib_fa},
        "feature_importance": {k: float(v) for k, v in clf.feature_importances().items()},
        "_pr_curve": (prec, rec),
    }


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "steel_industry_data.csv"))
    ap.add_argument("--sim-cache", default=str(ROOT / "data" / "sim_output.pkl"))
    ap.add_argument("--force-resim", action="store_true")
    ap.add_argument("--train-days", type=int, default=50)
    ap.add_argument("--horizon-days", type=int, default=10)
    ap.add_argument("--val-every-days", type=int, default=14)
    ap.add_argument("--test-every-days", type=int, default=7)
    ap.add_argument("--max-false-alarms-per-day", type=float, default=2.0)
    ap.add_argument("--n-random-scenarios", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--fig-dir", default=str(ROOT / "figures"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    PPD = PERIODS_PER_DAY

    print(f"[1/6] Loading real dataset from {args.data}")
    df = load_steel_data(args.data)
    usage_raw = df.set_index("date")["Usage_kWh"].asfreq("15min")

    sim = load_or_run_simulation(usage_raw, args.sim_cache, args.force_resim)
    sim = sim.set_index("date")
    sim.index = pd.DatetimeIndex(sim.index).to_period("15min").to_timestamp()
    sim = sim.asfreq("15min")

    summary = station_summary(sim)
    summary.to_csv(out_dir / "station_summary.csv")
    bottleneck = identify_bottleneck(sim)
    print(f"      Bottleneck identified: {bottleneck} "
          f"(mean utilization = {summary.loc[bottleneck, 'mean_util']:.3f}, "
          f"mean wait = {summary.loc[bottleneck, 'mean_wait_s']:.1f}s)")

    # ------------------------------------------------------------------ sensitivity
    print("[3/6] Parameter-sensitivity analysis")
    sens_usage = usage_raw.iloc[: 8 * 7 * PPD] if args.quick else usage_raw
    sens = run_sensitivity(sens_usage, cache_dir / ("sensitivity_quick" if args.quick else "sensitivity"),
                           jobs=args.jobs, n_random=2 if args.quick else args.n_random_scenarios)
    sens_summary = summarize_sensitivity(sens)
    print(f"      Base momentary-bottleneck shares: "
          + ", ".join(f"{s} {v:.1%}" for s, v in sens_summary["base"]["momentary_shares"].items()))
    print(f"      Random scenarios with {bottleneck} as bottleneck: "
          f"{sens_summary['random_bottleneck_share'][bottleneck]:.0%}")

    # ------------------------------------------------------------------ forecasting
    print("[4/6] Rolling-origin forecasting evaluation")
    util = sim[f"util_{bottleneck}"]
    usage = sim["usage_kwh"]
    window, horizon = args.train_days * PPD, args.horizon_days * PPD
    if args.quick:
        val_origins = make_origins(util.index, "2018-03-01", "2018-03-15", 14, window, horizon)
        test_origins = make_origins(util.index, "2018-07-01", "2018-07-08", 7, window, horizon)
    else:
        val_origins = make_origins(util.index, "2018-03-01", "2018-06-20", args.val_every_days, window, horizon)
        test_origins = make_origins(util.index, "2018-07-01", "2018-12-31", args.test_every_days, window, horizon)
    print(f"      {len(val_origins)} validation origins, {len(test_origins)} test origins, "
          f"{len(MODEL_REGISTRY)} models, horizons {list(HORIZONS)}")
    ev = run_rolling_evaluation(util, usage, val_origins, test_origins, window, horizon,
                                cache_dir / "rolling", jobs=args.jobs, grid_limit=1 if args.quick else None)
    rows = []
    for label, per_model in ev["metrics"].items():
        for k, m in per_model.items():
            rows.append({"horizon": label, "model": ev["models"][k], **m,
                         **ev["intervals"][k][label]})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_dir / "rolling_metrics.csv", index=False)
    print(metrics_df.pivot(index="model", columns="horizon", values="MASE")[list(HORIZONS)].round(3).to_string())

    last_pos = ev["_test_origin_pos"][-1]
    sarima = MODEL_REGISTRY["sarima"](**ev["tuned_params"]["sarima"]).fit(util.iloc[last_pos - window + 1: last_pos + 1])
    resid = sarima.in_sample_residuals().iloc[PPD:]  # skip the first day (state initialisation)
    lb = acorr_ljungbox(resid, lags=[24, 96])
    ljung_box = {str(lag): {"Q": float(r.lb_stat), "p": float(r.lb_pvalue)} for lag, r in lb.iterrows()}

    # ------------------------------------------------------------------ alerts
    print(f"[5/6] Bottleneck alerts (budget {args.max_false_alarms_per_day:g} false alarms/day)")
    periods = ((util.index[0], "2018-04-30 23:45"), ("2018-05-01", "2018-06-30 23:45"), ("2018-07-01", util.index[-1]))
    alerts = {}
    for label, lead in ALERT_LEADS.items():
        a = evaluate_alerts(util, usage, periods, lead, args.max_false_alarms_per_day)
        alerts[label] = a
        t, b = a["test"], a["persistence_baseline"]
        print(f"      lead {label}: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
              f"FA/day={t['false_alarms_per_day']:.2f} | persistence R={b['recall']:.3f} "
              f"FA/day={b['false_alarms_per_day']:.2f} | matched R={a['matched_to_persistence']['recall']:.3f} "
              f"FA/day={a['matched_to_persistence']['false_alarms_per_day']:.2f}")

    # ------------------------------------------------------------------ outputs
    results = {
        "version": bottleneck_sim.__version__,
        "quick_mode": bool(args.quick),
        "dataset": {"rows": int(len(df)), "start": str(df["date"].min()), "end": str(df["date"].max())},
        "stations": {name: dict(cfg) for name, cfg in DEFAULT_STATIONS.items()},
        "station_summary": {s: {k: float(v) for k, v in row.items()} for s, row in summary.iterrows()},
        "bottleneck": bottleneck,
        "sensitivity": sens_summary,
        "forecasting": {**ev, "sarima_ljung_box": ljung_box,
                        "train_days": args.train_days, "horizon_days": args.horizon_days,
                        "horizons": {k: v for k, v in HORIZONS.items()}},
        "alerts": {"max_false_alarms_per_day": args.max_false_alarms_per_day,
                   "periods": {"train": [str(p) for p in periods[0]], "calibration": [str(p) for p in periods[1]],
                               "test": [str(p) for p in periods[2]]},
                   **alerts},
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(strip_private(results), indent=2, default=str))

    if args.no_figures:
        print("[6/6] Skipping figures (--no-figures)")
    else:
        print(f"[6/6] Writing figures to {args.fig_dir}")
        make_all_figures(args.fig_dir, df=df, sim=sim, summary=summary, bottleneck=bottleneck,
                         sensitivity=sens_summary, evaluation=ev, util=util, window=window,
                         alerts=alerts)

    default_summary = Path(args.summary).resolve() == (ROOT / "results" / "results.json").resolve()
    if default_summary and not args.quick:
        import update_readme  # scripts/ is on sys.path when this file is run as a script
        update_readme.main(args.summary, ROOT / "README.md")

    print(f"\nDone. Results written to {out_dir}/ and {args.summary}")


if __name__ == "__main__":
    main()
