"""
Discrete-event simulation (SimPy) of a 5-station production line driven by the real
energy-usage signal. Imported by step 2 and step 3. You do not run this file.
"""
import numpy as np
import pandas as pd
import simpy

DEFAULT_STATIONS = {
    "Furnace":   dict(mean=55.0, cv=0.35, servers=1),
    "Rolling":   dict(mean=38.0, cv=0.30, servers=1),
    "Cooling":   dict(mean=30.0, cv=0.25, servers=1),
    "Coating":   dict(mean=42.0, cv=0.30, servers=1),
    "Packaging": dict(mean=25.0, cv=0.20, servers=2),
}


def _binned_busy_time(starts, ends, n_intervals, interval_sec):
    """Busy server-seconds inside each reporting interval."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    if starts.size == 0:
        return np.zeros(n_intervals)
    times = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(starts.size), -np.ones(ends.size)])
    order = np.argsort(times, kind="stable")
    times, deltas = times[order], deltas[order]
    busy_servers = np.cumsum(deltas)
    cum = np.concatenate([[0.0], np.cumsum(busy_servers[:-1] * np.diff(times))])
    edges = np.arange(n_intervals + 1) * interval_sec
    return np.diff(np.interp(edges, times, cum, left=0.0))


def run_simulation(usage, stations=None, interval_sec=15 * 60,
                   base_rate_per_sec=1 / 40.0, seed=42, return_spans=False):
    """
    usage : pd.Series with a DatetimeIndex, one value per 15-minute interval.
    Returns a DataFrame with, per interval and per station:
      util_<s>  busy server-seconds / capacity
      wait_<s>  mean queueing delay of jobs that joined the queue in the interval
      queue_<s> jobs waiting at the end of the interval
      jobs_<s>  jobs that joined the queue in the interval
    """
    stations = stations or DEFAULT_STATIONS
    station_names = list(stations.keys())
    rng = np.random.default_rng(seed)

    u = usage.to_numpy(dtype=float)
    n = len(u)
    span = u.max() - u.min()
    norm = (u - u.min()) / span if span > 0 else np.zeros(n)
    arrival_mult = 0.05 + 0.95 * norm

    # arrivals: Poisson count per interval, uniform times inside the interval
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
    env.run()  # let jobs still in the system finish so their waits are recorded

    out = pd.DataFrame({"date": usage.index, "usage_kwh": u, "arrivals": counts})
    for s in station_names:
        busy = _binned_busy_time(svc_start[s], svc_end[s], n, interval_sec)
        out[f"util_{s}"] = np.clip(busy / (interval_sec * stations[s]["servers"]), 0.0, 1.0)
        bins = (np.asarray(wait_join[s]) // interval_sec).astype(int)
        keep = bins < n
        w_sum = np.bincount(bins[keep], weights=np.asarray(wait_len[s])[keep], minlength=n)
        w_cnt = np.bincount(bins[keep], minlength=n)
        out[f"wait_{s}"] = np.divide(w_sum, w_cnt, out=np.zeros(n), where=w_cnt > 0)
        out[f"queue_{s}"] = queue_len[s]
        out[f"jobs_{s}"] = w_cnt
    if return_spans:
        return out, {s: (np.asarray(svc_start[s]), np.asarray(svc_end[s])) for s in station_names}
    return out


def momentary_bottleneck_shares(spans, horizon_sec, step_sec=60.0):
    """Active-period method (Roser et al., 2001): share of time each station is the momentary bottleneck."""
    t = np.arange(0.0, horizon_sec, step_sec) + step_sec / 2
    names = list(spans)
    current = np.full((len(names), t.size), -1.0)
    for j, s in enumerate(names):
        starts, ends = spans[s]
        if len(starts) == 0:
            continue
        order = np.argsort(starts)
        starts, ends = starts[order], np.maximum.accumulate(ends[order])
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


def station_summary(sim_df, stations=None):
    stations = stations or list(DEFAULT_STATIONS.keys())
    rows = []
    for s in stations:
        rows.append({
            "station": s,
            "mean_util": sim_df[f"util_{s}"].mean(),
            "p95_util": sim_df[f"util_{s}"].quantile(0.95),
            "mean_wait_s": np.average(sim_df[f"wait_{s}"], weights=sim_df[f"jobs_{s}"])
            if sim_df[f"jobs_{s}"].sum() > 0 else 0.0,
            "mean_queue": sim_df[f"queue_{s}"].mean(),
            "max_queue": int(sim_df[f"queue_{s}"].max()),
        })
    return pd.DataFrame(rows).set_index("station").sort_values("mean_util", ascending=False)
