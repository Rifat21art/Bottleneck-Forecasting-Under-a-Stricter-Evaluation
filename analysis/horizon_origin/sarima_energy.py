"""
SARIMA with lagged energy: the ablation model for analysis/horizon_origin.

"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bottleneck_sim.forecasting.base import register_model  # noqa: E402
from bottleneck_sim.forecasting.sarima_model import SARIMAForecaster  # noqa: E402
from bottleneck_sim.utils.features import make_fourier, seasonal_lag  # noqa: E402


@register_model("sarima_energy")
class SARIMAEnergyForecaster(SARIMAForecaster):
    name = "SARIMA with lagged energy"

    @property
    def cache_key(self):
        # distinct from "sarima" so the per-origin fit cache never hands this model a plain SARIMA fit
        return ("sarima_energy", json.dumps([list(self.order), self.K_daily, self.K_weekly]))

    def _exog(self, idx):
        exog = make_fourier(idx, self.K_daily, self.K_weekly)
        exog["usage_lastweek"] = seasonal_lag(self._usage, idx, self._origin).to_numpy() / self._scale
        return exog

    def fit(self, train_series, exog_train=None, fit_cache=None, usage_series=None, **kwargs):
        assert usage_series is not None, "usage_series (energy up to the origin) is required"
        self._origin = train_series.index.max()
        self._usage = usage_series.loc[:self._origin]
        # scale from the training rows only, so no post-origin information enters
        lagged = seasonal_lag(self._usage, train_series.index, self._origin)
        # rows without a full week of energy history (only at the very start of a series) are
        # dropped, as the ML models drop rows with missing lags; the remaining rows stay contiguous
        first = lagged.first_valid_index()
        train_series, lagged = train_series.loc[first:], lagged.loc[first:]
        self._scale = float(lagged.max()) or 1.0
        # passing exog_train keeps the parent from storing this fit under the plain SARIMA cache key
        return super().fit(train_series, exog_train=self._exog(train_series.index), fit_cache=None)
