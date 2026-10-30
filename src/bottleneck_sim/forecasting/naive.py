"""v3.1.0 - Seasonal-naive benchmark: repeat the most recent fully observed week."""
import numpy as np
import pandas as pd

from .base import BottleneckForecaster, register_model
from ..utils.features import seasonal_lag


@register_model("seasonal_naive")
class SeasonalNaiveForecaster(BottleneckForecaster):
    name = "Seasonal naive (weekly)"

    def fit(self, train_series, exog_train=None, **kwargs):
        self._train_series = train_series
        self._origin = train_series.index.max()
        return self

    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        idx = index if index is not None else exog_test.index
        fc = seasonal_lag(self._train_series, idx, self._origin)
        return pd.Series(np.nan_to_num(fc.to_numpy()), index=idx)
