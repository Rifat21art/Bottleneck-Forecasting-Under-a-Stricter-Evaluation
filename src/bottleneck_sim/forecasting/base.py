"""
Model registry for bottleneck-forecasting models.

To add a new model version:
  1. Create a new file in this package, e.g. `my_new_model.py`.
  2. Subclass `BottleneckForecaster` and implement `fit()` / `predict()`.
  3. Decorate the class with `@register_model("my_model_name")`.
  4. Add `from . import my_new_model` to `forecasting/__init__.py`.

That's it — the model is automatically picked up by `scripts/run_pipeline.py`
and included in the comparison table/plots. No other code needs to change.
See `ml_direct_model.py` for a minimal worked example (added in v2.1.0).

Tuning: a model may declare `param_grid`, a list of constructor-kwarg dicts
that `scripts/run_pipeline.py` searches on validation origins (default: the
constructor defaults only). `depends_on` maps a constructor kwarg to another
registered model whose tuned parameters should be injected into it (e.g. the
hybrid model reuses the tuned SARIMA configuration).

Information contract: `fit()` receives `usage_series`, the real exogenous
energy signal, truncated at the end of the training window. Models must not
assume any exogenous values after the forecast origin are known (see
`utils.features.seasonal_lag` for a valid ex-ante substitute).
"""
from abc import ABC, abstractmethod

MODEL_REGISTRY = {}


def register_model(name):
    """Class decorator that registers a BottleneckForecaster subclass under `name`."""
    def _wrap(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return _wrap


class BottleneckForecaster(ABC):
    """Common interface every forecasting model must implement."""

    name: str = "unnamed"
    param_grid: list = [{}]
    depends_on: dict = {}

    @abstractmethod
    def fit(self, train_series, exog_train=None, usage_series=None, **kwargs):
        """
        Fit on the training series (pandas Series, DatetimeIndex).

        `usage_series` holds the real exogenous signal up to the forecast
        origin only; models that don't use it should simply ignore it.
        `fit_cache` (optional dict, via kwargs) is shared by all models fitted
        on the same training series and may be used to reuse expensive fits.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, n_periods, exog_test=None, index=None, **kwargs):
        """Return a pandas Series of length n_periods indexed by `index` (the test timestamps)."""
        raise NotImplementedError

    def fit_predict(self, train_series, n_periods, exog_train=None, exog_test=None,
                    index=None, **kwargs):
        self.fit(train_series, exog_train=exog_train, **kwargs)
        return self.predict(n_periods, exog_test=exog_test, index=index)
