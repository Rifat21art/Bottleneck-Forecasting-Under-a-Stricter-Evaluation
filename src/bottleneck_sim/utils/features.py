"""Shared calendar / Fourier / exogenous feature builders used by multiple forecasters."""
import numpy as np
import pandas as pd

PERIODS_PER_DAY = 96
PERIODS_PER_WEEK = 7 * PERIODS_PER_DAY


def make_fourier(idx, K_daily=4, K_weekly=2):
    """Daily + weekly Fourier-pair regressors for 15-minute-resolution data."""
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


def seasonal_lag(series, idx, origin, period=PERIODS_PER_WEEK, freq="15min"):
    """
    Value of `series` at the same slot `k` seasons earlier, using the smallest
    k >= 1 that is not after the forecast `origin`.

    For timestamps at or before `origin` (training rows) this is simply
    lag-`period`; for timestamps h steps past `origin` it is lag `k*period`
    with k = ceil(h / period). The result therefore never uses information
    from after `origin`, so it is a valid ex-ante feature at any horizon.
    """
    idx = pd.DatetimeIndex(idx).as_unit("ns")  # pandas 3 may default to us; keep units consistent
    step_ns = pd.Timedelta(freq).value
    h = (idx.asi8 - pd.Timestamp(origin).as_unit("ns").value) / step_ns
    k = np.maximum(1, np.ceil(h / period)).astype(np.int64)
    src = idx - pd.to_timedelta(k * period * step_ns, unit="ns")
    return pd.Series(series.reindex(src).to_numpy(), index=idx)
