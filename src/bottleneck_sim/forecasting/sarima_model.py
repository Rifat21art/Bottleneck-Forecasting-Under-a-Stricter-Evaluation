"""
v1.0.0 - Seasonal model formulated as dynamic harmonic regression: ARMA errors
around Fourier daily/weekly seasonal terms. This is the standard fast
alternative to full seasonal ARIMA for high-frequency (15-min, m=96) data
(Hyndman & Athanasopoulos, 2021) and is what the rest of the codebase refers
to as "SARIMA".
"""
import json

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BottleneckForecaster, register_model
from ..utils.features import make_fourier


@register_model("sarima")
class SARIMAForecaster(BottleneckForecaster):
    name = "SARIMA"
    param_grid = [
        dict(order=order, K_daily=k)
        for order in [(2, 1, 2), (2, 0, 1)]
        for k in [4, 8]
    ]

    def __init__(self, order=(2, 1, 2), K_daily=4, K_weekly=2):
        self.order = tuple(order)
        self.K_daily, self.K_weekly = K_daily, K_weekly
        self.result_ = None

    @property
    def cache_key(self):
        return ("sarima", json.dumps([list(self.order), self.K_daily, self.K_weekly]))

    def _exog(self, idx):
        return make_fourier(idx, self.K_daily, self.K_weekly)

    def fit(self, train_series, exog_train=None, fit_cache=None, **kwargs):
        exog = exog_train if exog_train is not None else self._exog(train_series.index)
        self.result_ = SARIMAX(
            train_series, order=self.order, exog=exog,
            trend="c" if self.order[1] == 0 else None,  # undifferenced errors need a level
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=100, method="lbfgs")
        self.train_series_ = train_series
        if fit_cache is not None and exog_train is None:
            fit_cache[self.cache_key] = self
        return self

    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        exog = exog_test if exog_test is not None else self._exog(index)
        fc = self.result_.get_forecast(steps=n_periods, exog=exog).predicted_mean
        return fc.clip(0, 1)

    def in_sample_residuals(self):
        fitted = self.result_.fittedvalues.clip(0, 1)
        return self.train_series_ - fitted
