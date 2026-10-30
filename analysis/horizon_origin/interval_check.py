#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis"))

import pandas as pd  # noqa: E402

try:
    import pmdarima  # noqa: F401
except ImportError:
    # Every forecast used here is read from results/cache/rolling, so ARIMA is never refit.
    # A stub lets the model registry import; any cache miss fails loudly instead of refitting.
    import types
    def _no_fit(*args, **kwargs):
        raise RuntimeError("pmdarima is not installed and a forecast was missing from the cache")
    sys.modules["pmdarima"] = types.SimpleNamespace(auto_arima=_no_fit)

import bottleneck_sim.evaluation as ev  # noqa: E402
import shared as da  # noqa: E402

OUT = Path(__file__).with_name("interval_results.json")
CURRENT_BOUNDS = ev.interval_bounds


def pre_fix_bounds(forecast, quantiles):
    """The construction used before the fix: error quantiles added to the forecast."""
    lo, hi = np.empty_like(forecast), np.empty_like(forecast)
    for (a, b), (ql, qh) in zip(ev.INTERVAL_BUCKETS, quantiles):
        lo[..., a - 1:b] = forecast[..., a - 1:b] + ql
        hi[..., a - 1:b] = forecast[..., a - 1:b] + qh
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def run(origins, jobs=1):
    sim = da.load_sim()
    util, usage = sim["util_Furnace"], sim["usage_kwh"]
    out = {}
    for label, fn in (("pre_fix", pre_fix_bounds), ("corrected", CURRENT_BOUNDS)):
        ev.interval_bounds = fn
        res = da.evaluate(util, usage, origins(util), jobs)
        out[label] = res["intervals"]
        out["quantiles"] = res["interval_quantiles"]
    ev.interval_bounds = CURRENT_BOUNDS
    return out


def fixed_origins(util):
    return ev.make_origins(util.index, "2018-07-01", "2018-12-31", 7, da.WINDOW, da.HORIZON)


def phase_origins(util):
    first = util.index.get_indexer([pd.Timestamp("2018-07-01") - pd.Timedelta("15min")])[0]
    rng = np.random.default_rng(da.PHASE_SEED)
    return sorted(int(p) for p in rng.choice(np.arange(first, len(util) - da.HORIZON - 1), da.N_PHASE, replace=False))


def main():
    res = {"fixed": run(fixed_origins), "phase": run(phase_origins)}
    published = json.loads((ROOT / "results" / "results.json").read_text())["forecasting"]["intervals"]
    match = all(abs(res["fixed"]["corrected"][k][h]["coverage"] - published[k][h]["coverage"]) < 1e-12
                for k in published for h in published[k])
    res["corrected_reproduces_results_json"] = bool(match)
    OUT.write_text(json.dumps(res, indent=1))
    print("corrected construction reproduces results/results.json:", match)
    for design in ("fixed", "phase"):
        for k in res[design]["corrected"]:
            row = " ".join(f"{h}: {100 * res[design]['pre_fix'][k][h]['coverage']:.1f} -> "
                           f"{100 * res[design]['corrected'][k][h]['coverage']:.1f} (w {res[design]['corrected'][k][h]['mean_width']:.3f})"
                           for h in ev.HORIZONS)
            print(f"{design:5s} {k:18s} {row}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
