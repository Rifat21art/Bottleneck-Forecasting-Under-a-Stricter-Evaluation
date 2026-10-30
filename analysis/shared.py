"""Helpers shared for collaboration."""
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bottleneck_sim.evaluation import make_origins, run_rolling_evaluation  # noqa: E402
from bottleneck_sim.utils.features import PERIODS_PER_DAY as PPD  # noqa: E402

WINDOW, HORIZON = 50 * PPD, 10 * PPD
PHASE_SEED, N_PHASE = 7, 50


def load_sim():
    sim = pd.read_pickle(ROOT / "data" / "sim_output.pkl").set_index("date")
    sim.index = pd.DatetimeIndex(sim.index).to_period("15min").to_timestamp()
    return sim.asfreq("15min")


def evaluate(util, usage, test_origins, jobs):
    val = make_origins(util.index, "2018-03-01", "2018-06-20", 14, WINDOW, HORIZON)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_rolling_evaluation(util, usage, val, test_origins, WINDOW, HORIZON,
                                      ROOT / "results" / "cache" / "rolling", jobs=jobs)
