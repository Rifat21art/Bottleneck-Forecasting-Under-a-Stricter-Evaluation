import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from bottleneck_sim.simulation import (_binned_busy_time, run_simulation, station_summary,
                                       identify_bottleneck, SIM_VERSION)

STATIONS = [f"util_{s}" for s in ["Furnace", "Rolling", "Cooling", "Coating", "Packaging"]]


def _usage(values):
    idx = pd.date_range("2018-01-01", periods=len(values), freq="15min")
    return pd.Series(values, index=idx, dtype=float)


def test_busy_time_split_across_interval_boundaries():
    # [800,1000] straddles the first boundary; [850,1900] spans three intervals
    busy = _binned_busy_time([800, 850], [1000, 1900], n_intervals=3, interval_sec=900)
    assert np.allclose(busy, [150, 1000, 100])


@pytest.fixture(scope="module")
def overload_then_idle():
    # one day at peak demand (arrival rate > Furnace capacity), then two idle days
    return run_simulation(_usage(np.r_[np.full(96, 100.0), np.zeros(192)]))


def test_utilization_is_within_capacity(overload_then_idle):
    util = overload_then_idle[STATIONS]
    assert (util >= 0).all().all() and (util <= 1).all().all()


def test_backlog_work_is_credited_after_arrivals_stop(overload_then_idle):
    sim = overload_then_idle
    # arrivals drop to the 5% floor after day 1, but the Furnace keeps clearing its backlog
    assert sim["util_Furnace"].iloc[96:120].min() > 0.99
    assert sim["queue_Furnace"].iloc[96] > 0


def test_all_work_is_accounted_for(overload_then_idle):
    sim = overload_then_idle
    assert sim["queue_Furnace"].iloc[-1] == 0  # line has drained within the horizon
    for s in ["Furnace", "Packaging"]:
        assert sim[f"jobs_{s}"].sum() == sim["arrivals"].sum()
    busy = (sim["util_Furnace"] * 900).sum()
    assert busy == pytest.approx(sim["arrivals"].sum() * 55.0, rel=0.05)  # mean service time 55s


def test_summary_and_bottleneck(overload_then_idle):
    summary = station_summary(overload_then_idle)
    assert identify_bottleneck(overload_then_idle) == "Furnace" == summary.index[0]
    assert overload_then_idle.attrs["sim_version"] == SIM_VERSION


def test_simulation_is_deterministic_for_seed():
    usage = _usage(np.linspace(0, 50, 96))
    a, b = run_simulation(usage, seed=7), run_simulation(usage, seed=7)
    pd.testing.assert_frame_equal(a, b)
