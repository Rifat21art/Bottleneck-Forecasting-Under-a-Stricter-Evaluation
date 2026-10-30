from .base import MODEL_REGISTRY, register_model, BottleneckForecaster
# Import model modules so their @register_model decorators run and populate the registry.
# Adding a new model = write one file here + import it on this line. Nothing else changes.
from . import naive, arima_model, sarima_model, hybrid_sarima_xgb, ml_direct_model  # noqa: F401

__all__ = ["MODEL_REGISTRY", "register_model", "BottleneckForecaster"]
