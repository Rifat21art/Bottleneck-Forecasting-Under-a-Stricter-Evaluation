"""
Robustness of the bottleneck identification to the assumed (not fitted) station
parameters. Runs the full-year simulation under perturbed mean service times:

  - one-at-a-time: each station's mean service time scaled by 0.8 and 1.2;
  - joint: `n_random` scenarios with every station's mean independently scaled
    by U(0.8, 1.2).

For each scenario it records the utilization ranking and the active-period
momentary-bottleneck shares (Roser et al., 2001).
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .simulation import (DEFAULT_STATIONS, SIM_VERSION, run_simulation, station_summary,
                         momentary_bottleneck_shares)


def make_scenarios(stations=None, scale=0.2, n_random=30, seed=2024):
    stations = stations or DEFAULT_STATIONS
    scenarios = [{"id": "base", "kind": "base", "scales": {s: 1.0 for s in stations}}]
    for s in stations:
        for f in (1 - scale, 1 + scale):
            scales = {t: 1.0 for t in stations}
            scales[s] = f
            scenarios.append({"id": f"{s}x{f:.2f}", "kind": "one_at_a_time", "scales": scales})
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        scales = {s: float(rng.uniform(1 - scale, 1 + scale)) for s in stations}
        scenarios.append({"id": f"random{i:02d}", "kind": "random", "scales": scales})
    return scenarios


def _run_scenario(usage, scenario, interval_sec=15 * 60):
    stations = {s: {**cfg, "mean": cfg["mean"] * scenario["scales"][s]}
                for s, cfg in DEFAULT_STATIONS.items()}
    sim, spans = run_simulation(usage, stations=stations, return_spans=True)
    summary = station_summary(sim, list(stations))
    ranked = summary["mean_util"].sort_values(ascending=False)
    return {
        **scenario,
        "bottleneck": ranked.index[0],
        "runner_up": ranked.index[1],
        "util_margin": float(ranked.iloc[0] - ranked.iloc[1]),
        "mean_util": {s: float(v) for s, v in summary["mean_util"].items()},
        "mean_wait_s": {s: float(v) for s, v in summary["mean_wait_s"].items()},
        "momentary_shares": momentary_bottleneck_shares(spans, len(usage) * interval_sec),
    }


def run_sensitivity(usage, cache_dir, jobs=1, **scenario_kwargs):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_key = hashlib.sha1(np.ascontiguousarray(usage.to_numpy(dtype=float)).tobytes()).hexdigest()[:12]
    results, todo = [], []
    for sc in make_scenarios(**scenario_kwargs):
        key = hashlib.sha1(json.dumps([SIM_VERSION, data_key, sc], sort_keys=True).encode()).hexdigest()[:16]
        path = cache_dir / f"{sc['id']}_{key}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            todo.append((sc, path))
    if todo:
        print(f"    simulating {len(todo)} sensitivity scenarios ({len(results)} cached) on {jobs} worker(s)")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_run_scenario, usage, sc): path for sc, path in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                futures[fut].write_text(json.dumps(res))
                results.append(res)
                print(f"      [{n}/{len(todo)}] {res['id']}: bottleneck={res['bottleneck']} "
                      f"(margin {res['util_margin']:.3f})", flush=True)
    order = {sc["id"]: i for i, sc in enumerate(make_scenarios(**scenario_kwargs))}
    return sorted(results, key=lambda r: order[r["id"]])


def summarize_sensitivity(results):
    stations = list(results[0]["mean_util"])
    rand = [r for r in results if r["kind"] == "random"]
    base = next(r for r in results if r["kind"] == "base")
    return {
        "n_random": len(rand),
        "base": base,
        "one_at_a_time": [r for r in results if r["kind"] == "one_at_a_time"],
        "random_bottleneck_share": {s: sum(r["bottleneck"] == s for r in rand) / max(1, len(rand))
                                    for s in stations},
        "random_min_margin": min((r["util_margin"] for r in rand), default=float("nan")),
        "random_mean_momentary_shares": {s: float(np.mean([r["momentary_shares"][s] for r in rand]))
                                         for s in stations} if rand else {},
    }
