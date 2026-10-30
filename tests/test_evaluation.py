import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from bottleneck_sim.evaluation import (dm_test, holm, interval_bounds, interval_quantiles, interval_performance,
                                       make_origins, HORIZONS)
from bottleneck_sim.forecasting.classifier import choose_cutoff, alert_metrics
from bottleneck_sim.simulation import momentary_bottleneck_shares


def test_dm_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.1, 40)
    better = dm_test(base - 0.2, base + rng.normal(0, 0.05, 40))
    assert better["stat"] < 0 and better["p"] < 0.001
    same = dm_test(base, base + rng.normal(0, 0.05, 40))
    assert same["p"] > 0.05


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_intervals_reach_nominal_coverage_on_exchangeable_errors():
    rng = np.random.default_rng(1)
    horizon = HORIZONS["10d"]
    val_err = rng.normal(0, 0.1, (40, horizon))
    q = interval_quantiles(val_err, alpha=0.10)
    actual = np.full((40, horizon), 0.5) + rng.normal(0, 0.1, (40, horizon))
    perf = interval_performance(actual, np.full((40, horizon), 0.5), q)
    assert 0.87 < perf["10d"]["coverage"] < 0.93


def test_intervals_handle_skewed_errors():
    # Forecasts that tend to undershoot leave long negative tails in e = forecast - actual.
    # Symmetric errors cannot reveal a sign error in the interval; skewed ones can.
    rng = np.random.default_rng(2)
    horizon = HORIZONS["10d"]
    draw = lambda n: 0.05 - rng.exponential(0.1, (n, horizon))  # e = f - y, skewed negative
    q = interval_quantiles(draw(40), alpha=0.10)
    forecast = np.full((40, horizon), 0.4)
    perf = interval_performance(forecast - draw(40), forecast, q)
    assert 0.87 < perf["10d"]["coverage"] < 0.93
    lo, hi = interval_bounds(forecast, q)
    assert (hi - forecast).mean() > (forecast - lo).mean()  # undershooting forecasts need a longer upper arm


def test_origins_respect_history_and_horizon():
    idx = pd.date_range("2018-01-01", "2018-12-31 23:45", freq="15min")
    window, horizon = 50 * 96, 10 * 96
    pos = make_origins(idx, "2018-01-10", "2018-12-31", 7, window, horizon)
    assert all(p >= window + 672 and p + horizon < len(idx) for p in pos)
    assert all(idx[p + 1].hour == 0 and idx[p + 1].minute == 0 for p in pos)


def test_cutoff_respects_false_alarm_budget():
    rng = np.random.default_rng(2)
    y = (rng.random(96 * 20) < 0.2).astype(int)
    proba = np.clip(0.5 * y + rng.normal(0.25, 0.2, y.size), 0, 1)
    for budget in (0.5, 2, 10):
        cut = choose_cutoff(proba, y, budget)
        m = alert_metrics(y, (proba >= cut).astype(int))
        assert m["false_alarms_per_day"] <= budget + 1e-9
    # a looser budget can only raise recall
    recalls = [alert_metrics(y, (proba >= choose_cutoff(proba, y, b)).astype(int))["recall"] for b in (0.5, 2, 10)]
    assert recalls == sorted(recalls)


def test_momentary_bottleneck_prefers_longest_active_period():
    spans = {
        "A": (np.array([0.0]), np.array([100.0])),               # one long active period
        "B": (np.array([10.0, 30.0]), np.array([20.0, 40.0])),   # short periods inside A's
        "C": (np.array([150.0]), np.array([160.0])),             # alone later
    }
    shares = momentary_bottleneck_shares(spans, horizon_sec=200, step_sec=1)
    assert shares["A"] == pytest.approx(100 / 110)
    assert shares["B"] == 0.0
    assert shares["C"] == pytest.approx(10 / 110)
