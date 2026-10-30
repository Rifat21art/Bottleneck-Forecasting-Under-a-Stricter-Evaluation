"""v1.0.0 - Non-seasonal ARIMA baseline (pmdarima auto_arima, AIC stepwise search)."""
import numpy as np
import pandas as pd
import pmdarima as pm

from .base import BottleneckForecaster, register_model


@register_model("arima")
class ARIMAForecaster(BottleneckForecaster):
    name = "ARIMA"

    def __init__(self, max_p=3, max_q=3, max_d=1):
        self.max_p, self.max_q, self.max_d = max_p, max_q, max_d
        self.model_ = None

    def fit(self, train_series, exog_train=None, **kwargs):
        self.model_ = pm.auto_arima(
            train_series, seasonal=False, max_p=self.max_p, max_q=self.max_q,
            max_d=self.max_d, stepwise=True, suppress_warnings=True, error_action="ignore",
        )
        self.train_index_ = train_series.index
        return self

    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        fc = np.asarray(self.model_.predict(n_periods=n_periods))
        if index is not None:
            idx = index
        elif exog_test is not None:
            idx = exog_test.index
        else:
            idx = range(n_periods)
        return pd.Series(fc, index=idx).clip(0, 1)
