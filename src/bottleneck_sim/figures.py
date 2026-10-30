"""
Generates figures for the finals
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

DPI = 150
HIGHLIGHT, BASE = "#C00000", "#4472C4"


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(Path(out_dir) / name, dpi=DPI)
    plt.close(fig)


def data_figures(df, out_dir):
    usage = df.set_index("date")["Usage_kWh"]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(usage.index, usage.values, lw=0.3, color=BASE)
    ax.set_title("Steel Plant Energy Usage, 2018 (15-minute resolution)")
    ax.set_ylabel("Usage (kWh)")
    _save(fig, out_dir, "fig1_full_year_usage.png")

    wk = usage.loc["2018-03-05":"2018-03-11"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(wk.index, wk.values, lw=0.9, color=BASE)
    ax.set_title("One-Week Zoom: 5-11 March 2018")
    ax.set_ylabel("Usage (kWh)")
    _save(fig, out_dir, "fig2_week_zoom.png")

    counts = df["Load_Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(counts.index, counts.values, color=BASE)
    ax.set_title("Load Type Distribution")
    ax.set_ylabel("Observations")
    _save(fig, out_dir, "fig3_load_type_dist.png")

    hours = df["date"].dt.hour + df["date"].dt.minute / 60
    prof = df.assign(hour=hours).groupby(["WeekStatus", "hour"])["Usage_kWh"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for col in prof.columns:
        ax.plot(prof.index, prof[col], label=col, lw=1.5)
    ax.set_title("Average Daily Load Profile")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean usage (kWh)")
    ax.legend()
    _save(fig, out_dir, "fig4_daily_profile.png")


def simulation_figures(sim, summary, bottleneck, out_dir):
    stations = list(summary.index)
    order = [s for s in sim.columns if s.startswith("util_")]
    names = [c[len("util_"):] for c in order]
    colors = [HIGHLIGHT if s == bottleneck else BASE for s in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].bar(names, summary.loc[names, "mean_util"], color=colors)
    axes[0].set_title("Mean Station Utilization (Full Year)")
    axes[0].set_ylabel("Utilization")
    axes[1].bar(names, summary.loc[names, "mean_wait_s"], color=colors)
    axes[1].set_title("Mean Queueing Delay per Job (Full Year)")
    axes[1].set_ylabel("Wait time (s)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Simulation-Based Bottleneck Identification: {bottleneck} is the Constraint")
    _save(fig, out_dir, "fig_sim_bottleneck_id.png")

    roll = sim[order].rolling(30 * 96, min_periods=96).mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for col, name in zip(order, names):
        ax.plot(roll.index, roll[col], label=name, lw=2.2 if name == bottleneck else 1.0,
                color=HIGHLIGHT if name == bottleneck else None)
    ax.set_title("30-Day Rolling-Mean Utilization by Station")
    ax.set_ylabel("Utilization")
    ax.legend(ncol=len(stations), fontsize=8)
    _save(fig, out_dir, "fig_sim_utilization_trend.png")


def sensitivity_figures(sens, out_dir):
    rows = [sens["base"]] + sens["one_at_a_time"]
    stations = list(sens["base"]["mean_util"])
    mat = np.array([[r["mean_util"][s] for s in stations] for r in rows])
    labels = ["Base"] + [r["id"].replace("x0.80", " -20%").replace("x1.20", " +20%") for r in rows[1:]]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.imshow(mat, cmap="Blues", aspect="auto")
    for i, row in enumerate(mat):
        j_max = int(row.argmax())
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=HIGHLIGHT if j == j_max else ("white" if v > mat.max() * 0.6 else "black"),
                    fontweight="bold" if j == j_max else "normal")
    ax.set_xticks(range(len(stations)), stations, rotation=20, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("Mean utilization, one-at-a-time +/-20%\n(red = bottleneck)", fontsize=9)

    ax = axes[1]
    x = np.arange(len(stations))
    base_sh = [sens["base"]["momentary_shares"][s] for s in stations]
    rand_sh = [sens["random_mean_momentary_shares"].get(s, np.nan) for s in stations]
    ax.bar(x - 0.2, base_sh, width=0.4, label="Base parameters", color=BASE)
    ax.bar(x + 0.2, rand_sh, width=0.4, label=f"Mean of {sens['n_random']} random scenarios", color="#ED7D31")
    ax.set_xticks(x, stations, rotation=20, fontsize=8)
    ax.set_ylabel("Share of active time")
    ax.set_title("Momentary bottleneck share\n(active-period method)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig_sensitivity.png")


def diagnostic_figures(train, bottleneck, out_dir):
    dec = seasonal_decompose(train, model="additive", period=96)
    fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)
    for ax, (label, s) in zip(axes, [("Observed", dec.observed), ("Trend", dec.trend),
                                     ("Seasonal (daily)", dec.seasonal), ("Residual", dec.resid)]):
        ax.plot(s.index, s.values, lw=0.6)
        ax.set_ylabel(label, fontsize=8)
    axes[0].set_title(f"Additive Decomposition of {bottleneck} Utilization (Training Window, Last Test Origin)")
    _save(fig, out_dir, "fig5_decomposition.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    plot_acf(train, lags=2 * 96, ax=axes[0], markersize=2)
    plot_pacf(train, lags=2 * 96, ax=axes[1], markersize=2, method="ywm")
    axes[0].set_title("ACF (lags up to 2 days)")
    axes[1].set_title("PACF (lags up to 2 days)")
    _save(fig, out_dir, "fig6_acf_pacf.png")


def best_model(ev, horizon="10d"):
    return min(ev["metrics"][horizon], key=lambda k: ev["metrics"][horizon][k]["MASE"])


def forecast_figures(ev, util, bottleneck, out_dir):
    from .evaluation import interval_bounds, HORIZONS

    pos = ev["_test_origin_pos"][-1]
    H = ev["horizon"]
    idx = util.index[pos + 1: pos + 1 + H]
    actual = ev["_test_actual"][-1]
    band_key = best_model(ev)
    lo, hi = interval_bounds(ev["_test_fc"][band_key][-1], ev["interval_quantiles"][band_key])

    def draw(ax, n):
        ax.fill_between(idx[:n], lo[:n], hi[:n], color="#C9D7EE", label=f"90% interval ({ev['models'][band_key]})")
        ax.plot(idx[:n], actual[:n], color="black", lw=1.0, label="Actual (simulated)")
        for k, fc in ev["_test_fc"].items():
            ax.plot(idx[:n], fc[-1][:n], lw=1.1, alpha=0.9, label=ev["models"][k])

    fig, ax = plt.subplots(figsize=(11, 4))
    draw(ax, H)
    ax.set_title(f"{bottleneck} Utilization: {H // 96}-Day Forecasts from the Last Test Origin ({idx[0].date()})")
    ax.legend(fontsize=7, ncol=3)
    _save(fig, out_dir, "fig7_forecast_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    draw(ax, 96)
    ax.set_title("First 24 Hours of the Same Forecasts")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, "fig8_forecast_zoom.png")

    labels = list(HORIZONS)
    keys = list(ev["models"])
    width = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, k in enumerate(keys):
        vals = [ev["metrics"][h][k]["MASE"] for h in labels]
        ax.bar(np.arange(len(labels)) + (i - (len(keys) - 1) / 2) * width, vals, width=width, label=ev["models"][k])
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(labels)), [f"{h} ahead" for h in labels])
    ax.set_ylabel("MASE (mean over test origins)")
    ax.set_title("Forecast Error by Horizon (lower is better; <1 beats daily seasonal-naive)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, "fig9_horizon_mase.png")


def classifier_figures(alerts, out_dir, lead_label="1h"):
    a = alerts[lead_label]
    cm = np.asarray(a["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["No alert", "Alert"])
    ax.set_yticks([0, 1], ["No alert", "Alert"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Alert Confusion Matrix ({lead_label} lead, test)")
    _save(fig, out_dir, "fig10_confusion_matrix.png")

    imp = pd.Series(a["feature_importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(imp.index, imp.values, color=BASE)
    ax.set_title(f"Alert Classifier Feature Importance ({lead_label} lead, gain)")
    _save(fig, out_dir, "fig11_feature_importance.png")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for (label, al), color in zip(alerts.items(), [BASE, "#ED7D31"]):
        prec, rec = al["_pr_curve"]
        ax.plot(rec, prec, color=color, lw=1.3, label=f"XGBoost, {label} lead")
        t, b = al["test"], al["persistence_baseline"]
        ax.scatter([t["recall"]], [t["precision"]], color=color, zorder=3, s=40, edgecolor="black")
        ax.scatter([b["recall"]], [b["precision"]], color=color, marker="x", zorder=3, s=50)
    ax.scatter([], [], color="grey", edgecolor="black", label="Chosen operating point")
    ax.scatter([], [], color="grey", marker="x", label="Persistence rule (alert if busy now)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Bottleneck-Alert Precision-Recall (test period)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "fig12_alert_pr.png")


def make_all_figures(out_dir, *, df, sim, summary, bottleneck, sensitivity, evaluation, util, window, alerts):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    data_figures(df, out_dir)
    simulation_figures(sim, summary, bottleneck, out_dir)
    sensitivity_figures(sensitivity, out_dir)
    pos = evaluation["_test_origin_pos"][-1]
    diagnostic_figures(util.iloc[pos - window + 1: pos + 1], bottleneck, out_dir)
    forecast_figures(evaluation, util, bottleneck, out_dir)
    classifier_figures(alerts, out_dir)
