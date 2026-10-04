"""
The five forecasting models as plain functions (no registry, no classes).
Imported by step 4, step 5 and step 6. You do not run this file.

Every function forecasts for the timestamps in `idx` using only data up to the forecast origin.
"""
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from common import make_fourier, make_calendar, seasonal_lag

MODEL_NAMES = {
    "seasonal_naive": "Seasonal naive (weekly)",
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "hybrid_sarima_xgb": "Hybrid SARIMA+XGB",
    "ml_direct_xgb": "ML-Direct (XGBoost, recursive)",
}

# ------------------------------------------------------------------ tuning grids
SARIMA_GRID = [dict(order=order, K_daily=k) for order in [(2, 1, 2), (2, 0, 1)] for k in [4, 8]]

_HYBRID_BASE = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)
HYBRID_GRID = [{**_HYBRID_BASE, "max_depth": d, "n_estimators": n} for d in [3, 5] for n in [150, 300]]

_ML_BASE = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42)
ML_GRID = [{**_ML_BASE, "max_depth": d, "n_estimators": n} for d in [3, 5, 7] for n in [200, 400]]

ML_LAGS = (1, 2, 3, 4, 96, 672)


# ------------------------------------------------------------------ 1. seasonal naive
def seasonal_naive_forecast(train, idx):
    """Repeat the most recent fully observed week."""
    fc = seasonal_lag(train, idx, train.index.max())
    return np.nan_to_num(fc.to_numpy())


# ------------------------------------------------------------------ 2. ARIMA
def arima_forecast(train, idx):
    model = pm.auto_arima(train, seasonal=False, max_p=3, max_q=3, max_d=1,
                          stepwise=True, suppress_warnings=True, error_action="ignore")
    return np.clip(np.asarray(model.predict(n_periods=len(idx))), 0, 1)


# ------------------------------------------------------------------ 3. SARIMA (Fourier terms + ARMA errors)
def sarima_fit(train, order=(2, 1, 2), K_daily=4, K_weekly=2):
    order = tuple(order)
    exog = make_fourier(train.index, K_daily, K_weekly)
    return SARIMAX(train, order=order, exog=exog,
                   trend="c" if order[1] == 0 else None,
                   enforce_stationarity=False, enforce_invertibility=False
                   ).fit(disp=False, maxiter=100, method="lbfgs")


def sarima_forecast(result, idx, K_daily=4, K_weekly=2):
    exog = make_fourier(pd.DatetimeIndex(idx), K_daily, K_weekly)
    fc = result.get_forecast(steps=len(idx), exog=exog).predicted_mean
    return np.clip(np.asarray(fc), 0, 1)


def sarima_residuals(result, train):
    return train - result.fittedvalues.clip(0, 1)


# ------------------------------------------------------------------ 4. hybrid SARIMA + XGBoost on residuals
def hybrid_forecast(sarima_result, train, usage_train, idx, K_daily, xgb_kwargs, K_weekly=2):
    """SARIMA forecast + XGBoost correction trained on SARIMA's own in-sample residuals."""
    idx = pd.DatetimeIndex(idx)
    origin = train.index.max()
    usage = usage_train.loc[:origin]
    resid = sarima_residuals(sarima_result, train)

    def feats(ix):
        f = make_calendar(ix)
        f["usage_lastweek"] = seasonal_lag(usage, ix, origin).to_numpy()
        f["resid_lastweek"] = seasonal_lag(resid, ix, origin).to_numpy()
        return f

    xgb = XGBRegressor(**xgb_kwargs)
    xgb.fit(feats(train.index), resid.reindex(train.index))
    base = sarima_forecast(sarima_result, idx, K_daily, K_weekly)
    return np.clip(base + xgb.predict(feats(idx)), 0, 1)


# ------------------------------------------------------------------ 5. ML-direct XGBoost (recursive)
def ml_forecast(train, usage_train, idx, xgb_kwargs, lags=ML_LAGS):
    idx = pd.DatetimeIndex(idx)
    origin = train.index.max()
    usage = usage_train.loc[:origin]

    def exog_features(ix):
        f = make_calendar(ix)
        f["usage_lastweek"] = seasonal_lag(usage, ix, origin).to_numpy()
        return f

    feat = exog_features(train.index)
    for lag in lags:
        feat[f"lag{lag}"] = train.shift(lag).reindex(train.index)
    feat = feat.dropna()
    model = XGBRegressor(**xgb_kwargs)
    model.fit(feat, train.reindex(feat.index))

    # recursive prediction: each forecast becomes a lag feature for later steps
    exog = exog_features(idx).to_numpy(dtype=float)
    n_train = len(train)
    history = np.concatenate([train.to_numpy(dtype=float), np.full(len(idx), np.nan)])
    lag_arr = np.asarray(lags)
    booster = model.get_booster()
    row = np.empty((1, exog.shape[1] + len(lag_arr)))
    for i in range(len(idx)):
        row[0, :exog.shape[1]] = exog[i]
        row[0, exog.shape[1]:] = history[n_train + i - lag_arr]
        history[n_train + i] = min(max(float(booster.inplace_predict(row)[0]), 0.0), 1.0)
    return history[n_train:]
