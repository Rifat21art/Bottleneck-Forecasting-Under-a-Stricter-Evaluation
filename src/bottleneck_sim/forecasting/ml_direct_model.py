"""
v2.1.0 - Pure machine-learning forecaster (NEW, added purely as a registry
drop-in to demonstrate extensibility - see forecasting/base.py docstring).

No SARIMA component at all: a single XGBoost regressor is trained directly on
lagged-target + calendar + energy-usage features and forecasts recursively,
feeding each prediction back in as the next step's lag features. This is a
common, strong, purely-ML baseline to compare against the statistical and
hybrid models above.

Since v3.0.0 the usage feature is the value at the same slot in the most
recent fully observed week (`seasonal_lag`), so no future exogenous data is
used at forecast time.

Adding this file + one import line in forecasting/__init__.py was the entire
integration cost - fit/predict/evaluate/plots in run_pipeline.py needed no
changes.
"""
import numpy as np
from xgboost import XGBRegressor
import pandas as pd

from .base import BottleneckForecaster, register_model
from ..utils.features import make_calendar, seasonal_lag

DEFAULT_XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)


@register_model("ml_direct_xgb")
class MLDirectForecaster(BottleneckForecaster):
    name = "ML-Direct (XGBoost, recursive)"
    param_grid = [dict(xgb_kwargs={**DEFAULT_XGB, "max_depth": d, "n_estimators": n})
                  for d in [3, 5, 7] for n in [200, 400]]

    def __init__(self, lags=(1, 2, 3, 4, 96, 672), xgb_kwargs=None):
        self.lags = tuple(lags)
        self.xgb_kwargs = xgb_kwargs or dict(DEFAULT_XGB)
        self.model_ = None

    def _exog_features(self, idx):
        feat = make_calendar(idx)
        feat["usage_lastweek"] = seasonal_lag(self._usage_series, idx, self._origin).to_numpy()
        return feat

    def _features(self, idx, history):
        feat = self._exog_features(idx)
        for lag in self.lags:
            feat[f"lag{lag}"] = history.shift(lag).reindex(idx)
        return feat

    def fit(self, train_series, exog_train=None, usage_series=None, **kwargs):
        assert usage_series is not None, "usage_series (real exogenous signal) is required"
        self._origin = train_series.index.max()
        self._usage_series = usage_series.loc[:self._origin]
        self._train_series = train_series
        feat = self._features(train_series.index, train_series).dropna()
        y = train_series.reindex(feat.index)
        self.model_ = XGBRegressor(**self.xgb_kwargs)
        self.model_.fit(feat, y)
        return self

    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        # recursive multi-step forecast: each prediction becomes a lag feature for later
        # steps. Exogenous columns are built once; lag columns are read from a flat buffer
        # (column order matches `_features`: calendar, usage, then lags).
        idx = index if index is not None else exog_test.index
        exog = self._exog_features(pd.DatetimeIndex(idx)).to_numpy(dtype=float)
        n_train = len(self._train_series)
        history = np.concatenate([self._train_series.to_numpy(dtype=float), np.full(len(idx), np.nan)])
        lags = np.asarray(self.lags)
        booster = self.model_.get_booster()
        row = np.empty((1, exog.shape[1] + len(lags)))
        for i in range(len(idx)):
            row[0, :exog.shape[1]] = exog[i]
            row[0, exog.shape[1]:] = history[n_train + i - lags]
            history[n_train + i] = min(max(float(booster.inplace_predict(row)[0]), 0.0), 1.0)
        return pd.Series(history[n_train:], index=idx)
