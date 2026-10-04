"""
Shared helpers used by every step script (paths, data loading, features, metrics).
You do not run this file. The step scripts import it, so keep it in the same folder.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "steel_industry_data.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
CACHE = OUT / "forecast_cache"
for _p in (OUT, FIG, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ settings
PPD = 96                    # 15-minute periods per day
WEEK = 7 * PPD              # 672
WINDOW = 50 * PPD           # training window = 50 days
HORIZON = 10 * PPD          # forecast 10 days ahead
HORIZONS = {"1h": 4, "4h": 16, "1d": 96, "10d": 960}
INTERVAL_BUCKETS = [(1, 4), (5, 16), (17, 96), (97, 960)]   # 1-based horizon steps
MODEL_KEYS = ["seasonal_naive", "arima", "sarima", "hybrid_sarima_xgb", "ml_direct_xgb"]


# ------------------------------------------------------------------ data
def load_steel_data():
    df = pd.read_csv(DATA_CSV)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
    return df.sort_values("date").reset_index(drop=True)


def load_sim():
    """Simulation output written by step 2."""
    sim = pd.read_csv(OUT / "sim_output.csv", parse_dates=["date"])
    return sim.set_index("date").asfreq("15min")


def get_bottleneck():
    """Station with the highest mean utilization (from step 2's station_summary.csv)."""
    s = pd.read_csv(OUT / "station_summary.csv", index_col=0)
    return s["mean_util"].idxmax()


def cached(path, fn):
    """Tiny resume helper: load the .npy if it exists, else compute with fn(), save, return."""
    path = Path(path)
    if path.exists():
        return np.load(path)
    arr = np.asarray(fn(), dtype=float)
    np.save(path, arr)
    return arr


# ------------------------------------------------------------------ features
def make_fourier(idx, K_daily=4, K_weekly=2):
    t_day = (idx.hour * 3600 + idx.minute * 60) / 86400.0
    t_week = idx.dayofweek + t_day
    exog = pd.DataFrame(index=idx)
    for k in range(1, K_daily + 1):
        exog[f"day_sin{k}"] = np.sin(2 * np.pi * k * t_day)
        exog[f"day_cos{k}"] = np.cos(2 * np.pi * k * t_day)
    for k in range(1, K_weekly + 1):
        exog[f"week_sin{k}"] = np.sin(2 * np.pi * k * t_week / 7)
        exog[f"week_cos{k}"] = np.cos(2 * np.pi * k * t_week / 7)
    return exog


def make_calendar(idx):
    hour = idx.hour + idx.minute / 60
    exog = pd.DataFrame(index=idx)
    exog["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    exog["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    exog["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return exog


def seasonal_lag(series, idx, origin, period=WEEK, freq="15min"):
    """Value of `series` at the same slot k weeks earlier, smallest k>=1 that is not after `origin`.
    Never looks past the forecast origin, so it is a valid feature at any horizon."""
    idx = pd.DatetimeIndex(idx)
    step_ns = pd.Timedelta(freq).value
    h = (idx.asi8 - pd.Timestamp(origin).value) / step_ns
    k = np.maximum(1, np.ceil(h / period)).astype(np.int64)
    src = idx - pd.to_timedelta(k * period * step_ns, unit="ns")
    return pd.Series(series.reindex(src).to_numpy(), index=idx)


# ------------------------------------------------------------------ rolling origins
def make_origins(index, first_day, last_day, every_days, window=WINDOW, horizon=HORIZON):
    """Positions (row numbers) of the last row of each training window."""
    starts = pd.date_range(first_day, last_day, freq=f"{every_days}D")
    positions = []
    for day in starts:
        pos = index.get_indexer([day - pd.Timedelta("15min")])[0]
        if pos >= window + WEEK and pos + horizon < len(index):
            positions.append(int(pos))
    return positions


# ------------------------------------------------------------------ forecast metrics
def mase_scale(train, m=PPD):
    train = np.asarray(train, dtype=float)
    return float(np.mean(np.abs(train[m:] - train[:-m])))


def per_origin_errors(actual, forecast, scales):
    """dict horizon -> dict(MAE, MSE, MASE), each an array with one value per origin."""
    err = forecast - actual
    out = {}
    for label, H in HORIZONS.items():
        mae = np.abs(err[:, :H]).mean(axis=1)
        out[label] = {"MAE": mae, "MSE": (err[:, :H] ** 2).mean(axis=1), "MASE": mae / scales}
    return out


def summarize_errors(errs):
    return {label: {"MAE": float(e["MAE"].mean()), "RMSE": float(np.sqrt(e["MSE"].mean())),
                    "MASE": float(e["MASE"].mean()), "n_origins": int(e["MAE"].size)}
            for label, e in errs.items()}


def dm_test(loss_a, loss_b, lag=0):
    """Diebold-Mariano test (Harvey-Leybourne-Newbold correction). Negative stat => A better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = d.size
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(lag + 1)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if var <= 0:
        var = gamma[0]
    if var <= 0:
        return {"mean_diff": float(d.mean()), "stat": 0.0, "p": 1.0}
    h = lag + 1
    stat = d.mean() / math.sqrt(var / n)
    stat *= math.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(p)}


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def interval_quantiles(val_errors, alpha=0.10):
    """(low, high) error quantiles per horizon bucket, from validation errors (origins x horizon)."""
    q = []
    for a, b in INTERVAL_BUCKETS:
        e = np.sort(val_errors[:, a - 1:b].ravel())
        n = e.size
        k_hi = min(n - 1, math.ceil((n + 1) * (1 - alpha / 2)) - 1)
        k_lo = max(0, math.floor((n + 1) * (alpha / 2)) - 1)
        q.append((float(e[k_lo]), float(e[k_hi])))
    return q


def interval_bounds(forecast, quantiles):
    lo, hi = np.empty_like(forecast), np.empty_like(forecast)
    for (a, b), (ql, qh) in zip(INTERVAL_BUCKETS, quantiles):
        lo[..., a - 1:b] = forecast[..., a - 1:b] + ql
        hi[..., a - 1:b] = forecast[..., a - 1:b] + qh
    return np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def interval_performance(actual, forecast, quantiles):
    lo, hi = interval_bounds(forecast, quantiles)
    inside = (actual >= lo) & (actual <= hi)
    return {label: {"coverage": float(inside[:, :H].mean()), "mean_width": float((hi - lo)[:, :H].mean())}
            for label, H in HORIZONS.items()}
