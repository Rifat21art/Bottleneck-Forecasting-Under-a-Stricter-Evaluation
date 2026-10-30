"""
v2.0.0 - Hybrid forecaster: SARIMA point forecast + an XGBoost regressor
trained to predict SARIMA's own in-sample residuals. At forecast time the
XGBoost residual correction is added back to the SARIMA point forecast.

Every correction feature is one that is genuinely available at the forecast
origin for every horizon in the test window, and is built identically for
training and forecasting rows (v3.0.0):
  - calendar terms;
  - energy usage at the same slot in the most recent fully observed week
    (`seasonal_lag`), instead of the actual future usage;
  - SARIMA residual at the same slot in the most recent fully observed week,
    instead of short residual lags (1, 2 periods) that do not exist beyond
    the first forecast step.
"""
import pandas as pd
from xgboost import XGBRegressor

from .base import BottleneckForecaster, register_model
from .sarima_model import SARIMAForecaster
from ..utils.features import make_calendar, seasonal_lag

DEFAULT_XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42)


@register_model("hybrid_sarima_xgb")
class HybridSARIMAXGBForecaster(BottleneckForecaster):
    name = "Hybrid SARIMA+XGB"
    param_grid = [dict(xgb_kwargs={**DEFAULT_XGB, "max_depth": d, "n_estimators": n})
                  for d in [3, 5] for n in [150, 300]]
    depends_on = {"sarima_kwargs": "sarima"}  # use the tuned SARIMA configuration

    def __init__(self, sarima_kwargs=None, xgb_kwargs=None):
        self.sarima = SARIMAForecaster(**(sarima_kwargs or {}))
        self.xgb_kwargs = xgb_kwargs or dict(DEFAULT_XGB)
        self.xgb_ = None

    def _features(self, idx, origin):
        feat = make_calendar(idx)
        feat["usage_lastweek"] = seasonal_lag(self._usage_series, idx, origin).to_numpy()
        feat["resid_lastweek"] = seasonal_lag(self._resid_train, idx, origin).to_numpy()
        return feat  # NaNs (first training week) are handled natively by XGBoost

    def fit(self, train_series, exog_train=None, usage_series=None, fit_cache=None, **kwargs):
        assert usage_series is not None, "usage_series (real exogenous signal) is required"
        cached = fit_cache.get(self.sarima.cache_key) if fit_cache is not None and exog_train is None else None
        if cached is not None:
            self.sarima = cached
        else:
            self.sarima.fit(train_series, exog_train=exog_train, fit_cache=fit_cache)
        self._resid_train = self.sarima.in_sample_residuals()
        self._usage_series = usage_series.loc[:train_series.index.max()]
        self._origin = train_series.index.max()

        feat_train = self._features(train_series.index, self._origin)
        self.xgb_ = XGBRegressor(**self.xgb_kwargs)
        self.xgb_.fit(feat_train, self._resid_train.reindex(train_series.index))
        return self

    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        sarima_fc = self.sarima.predict(n_periods, exog_test=exog_test, index=index)
        correction = self.xgb_.predict(self._features(sarima_fc.index, self._origin))
        return pd.Series(sarima_fc.to_numpy() + correction, index=sarima_fc.index).clip(0, 1)
