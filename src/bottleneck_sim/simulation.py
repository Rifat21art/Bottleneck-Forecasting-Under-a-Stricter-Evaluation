"""
Discrete-event simulation of a 5-stage production line, driven by a real
exogenous demand-intensity signal.

Bookkeeping: every service span and every queue wait is recorded against
absolute simulation time. Busy time is then binned into reporting intervals
by overlap, so work that spills across an interval boundary (e.g. a backlog
built up during a demand peak) is credited to the interval in which it was
actually performed rather than being lost.
"""
import numpy as np
import pandas as pd
import simpy

SIM_VERSION = 2  # bump whenever simulation output semantics change (invalidates caches)

DEFAULT_STATIONS = {
    "Furnace":   dict(mean=55.0, cv=0.35, servers=1),
    "Rolling":   dict(mean=38.0, cv=0.30, servers=1),
    "Cooling":   dict(mean=30.0, cv=0.25, servers=1),
    "Coating":   dict(mean=42.0, cv=0.30, servers=1),
    "Packaging": dict(mean=25.0, cv=0.20, servers=2),
}


def _binned_busy_time(starts, ends, n_intervals, interval_sec):
    """Total busy server-seconds falling inside each interval [i*dt, (i+1)*dt)."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if starts.size == 0:
        return np.zeros(n_intervals)
    times = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(starts.size), -np.ones(ends.size)])
    order = np.argsort(times, kind="stable")
    times, deltas = times[order], deltas[order]
    busy_servers = np.cumsum(deltas)
    # cumulative busy server-seconds is piecewise linear between events
    cum = np.concatenate([[0.0], np.cumsum(busy_servers[:-1] * np.diff(times))])
    edges = np.arange(n_intervals + 1) * interval_sec
    return np.diff(np.interp(edges, times, cum, left=0.0))


def run_simulation(usage: pd.Series, stations=None, interval_sec=15 * 60,
                   base_rate_per_sec=1 / 40.0, seed=42, return_spans=False):
    """
    Run the multi-station DES over one row of `usage` per interval.

    Parameters
    ----------
    usage : pd.Series (DatetimeIndex) - real exogenous demand-intensity signal
            (e.g. plant energy usage), one value per interval.
    stations : dict of station configs, defaults to DEFAULT_STATIONS.
    interval_sec : real-world seconds represented by one row of `usage`.
    base_rate_per_sec : job-arrival rate at full (normalized=1) demand.
    return_spans : also return {station: (service_starts, service_ends)} in seconds.

    Returns
    -------
    pd.DataFrame with per-interval columns per station:
      util_<s>  - busy server-seconds performed in the interval / capacity
      wait_<s>  - mean queueing delay of jobs that joined the queue in the interval
      queue_<s> - number of jobs waiting at the end of the interval
      jobs_<s>  - number of jobs that joined the station's queue in the interval
    """
    stations = stations or DEFAULT_STATIONS
    station_names = list(stations.keys())
    rng = np.random.default_rng(seed)

    u = usage.to_numpy(dtype=float)
    n = len(u)
    span = u.max() - u.min()
    norm = (u - u.min()) / span if span > 0 else np.zeros(n)
    arrival_mult = 0.05 + 0.95 * norm

    # Non-homogeneous Poisson arrivals, piecewise-constant rate per interval:
    # given the count, arrival times are uniform within the interval.
    counts = rng.poisson(base_rate_per_sec * interval_sec * arrival_mult)
    offsets = rng.uniform(0.0, interval_sec, size=counts.sum())
    starts_of_interval = np.repeat(np.arange(n) * interval_sec, counts)
    arrival_times = np.sort(starts_of_interval + offsets)

    def sample_service_time(name):
        p = stations[name]
        shape = 1 / (p["cv"] ** 2)
        scale = p["mean"] / shape
        return max(1.0, rng.gamma(shape, scale))

    svc_start = {s: [] for s in station_names}
    svc_end = {s: [] for s in station_names}
    wait_join = {s: [] for s in station_names}
    wait_len = {s: [] for s in station_names}

    def job(env):
        for st in station_names:
            with resources[st].request() as req:
                joined = env.now
                yield req
                wait_join[st].append(joined)
                wait_len[st].append(env.now - joined)
                svc = sample_service_time(st)
                svc_start[st].append(env.now)
                yield env.timeout(svc)
                svc_end[st].append(env.now)

    def source(env):
        for t in arrival_times:
            yield env.timeout(t - env.now)
            env.process(job(env))

    env = simpy.Environment()
    resources = {name: simpy.Resource(env, capacity=p["servers"]) for name, p in stations.items()}
    env.process(source(env))

    queue_len = {s: np.zeros(n, dtype=int) for s in station_names}
    for i in range(n):
        env.run(until=(i + 1) * interval_sec)
        for s in station_names:
            queue_len[s][i] = len(resources[s].queue)
    env.run()  # drain: let jobs still in the system finish so their waits are recorded

    out = pd.DataFrame({"date": usage.index, "usage_kwh": u, "arrivals": counts})
    for s in station_names:
        busy = _binned_busy_time(svc_start[s], svc_end[s], n, interval_sec)
        # busy <= capacity by construction; clip only guards float round-off
        out[f"util_{s}"] = np.clip(busy / (interval_sec * stations[s]["servers"]), 0.0, 1.0)

        bins = (np.asarray(wait_join[s]) // interval_sec).astype(int)
        keep = bins < n
        w_sum = np.bincount(bins[keep], weights=np.asarray(wait_len[s])[keep], minlength=n)
        w_cnt = np.bincount(bins[keep], minlength=n)
        out[f"wait_{s}"] = np.divide(w_sum, w_cnt, out=np.zeros(n), where=w_cnt > 0)
        out[f"queue_{s}"] = queue_len[s]
        out[f"jobs_{s}"] = w_cnt
    out.attrs["sim_version"] = SIM_VERSION
    if return_spans:
        return out, {s: (np.asarray(svc_start[s]), np.asarray(svc_end[s])) for s in station_names}
    return out


def momentary_bottleneck_shares(spans, horizon_sec, step_sec=60.0):
    """
    Active-period method (Roser, Nakano & Tanaka, 2001): a station is active
    while it is working (here: while any of its servers is busy). At each time
    point the momentary bottleneck is the active station whose current active
    period is longest. Returns each station's share of sampled time points at
    which at least one station is active, sampled every `step_sec` seconds.
    """
    t = np.arange(0.0, horizon_sec, step_sec) + step_sec / 2
    names = list(spans)
    current = np.full((len(names), t.size), -1.0)
    for j, s in enumerate(names):
        starts, ends = spans[s]
        if len(starts) == 0:
            continue
        order = np.argsort(starts)
        starts, ends = starts[order], np.maximum.accumulate(ends[order])
        # merge overlapping/touching busy spans (parallel servers, back-to-back jobs)
        new_period = np.r_[True, starts[1:] > ends[:-1] + 1e-9]
        p_start = starts[new_period]
        p_end = np.maximum.reduceat(ends, np.flatnonzero(new_period))
        k = np.searchsorted(p_start, t, side="right") - 1
        active = (k >= 0) & (t < p_end[np.clip(k, 0, None)])
        current[j, active] = (p_end - p_start)[k[active]]
    any_active = current.max(axis=0) > 0
    winner = current[:, any_active].argmax(axis=0)
    counts = np.bincount(winner, minlength=len(names))
    return {s: float(c / max(1, any_active.sum())) for s, c in zip(names, counts)}


def station_summary(sim_df: pd.DataFrame, stations=None) -> pd.DataFrame:
    """Per-station mean utilization, job-weighted mean wait, and queue statistics."""
    stations = stations or list(DEFAULT_STATIONS.keys())
    rows = []
    for s in stations:
        rows.append({
            "station": s,
            "mean_util": sim_df[f"util_{s}"].mean(),
            "p95_util": sim_df[f"util_{s}"].quantile(0.95),
            # job-weighted: quiet intervals with no jobs must not dilute the mean delay
            "mean_wait_s": np.average(sim_df[f"wait_{s}"], weights=sim_df[f"jobs_{s}"])
            if sim_df[f"jobs_{s}"].sum() > 0 else 0.0,
            "mean_queue": sim_df[f"queue_{s}"].mean(),
            "max_queue": int(sim_df[f"queue_{s}"].max()),
        })
    return pd.DataFrame(rows).set_index("station").sort_values("mean_util", ascending=False)


def identify_bottleneck(sim_df: pd.DataFrame, stations=None) -> str:
    """Return the station name with the highest mean utilization."""
    return station_summary(sim_df, stations)["mean_util"].idxmax()
