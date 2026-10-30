import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from bottleneck_sim.utils.metrics import smape, mase, evaluate


def test_smape_zero_for_perfect_forecast():
    y = np.array([0.1, 0.5, 0.9])
    assert smape(y, y) == 0.0


def test_evaluate_returns_expected_keys():
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.18, 0.35])
    m = evaluate(y_true, y_pred)
    assert set(m.keys()) == {"MAE", "RMSE", "sMAPE"}
    assert m["MAE"] > 0
    assert "MASE" in evaluate(y_true, y_pred, y_train=np.arange(10.0), m=2)


def test_mase_equals_one_for_seasonal_naive_scale():
    y_train = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 2.0])  # seasonal-naive (m=2) in-sample MAE = 0.25
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.25, 0.75])                      # MAE = 0.25
    assert np.isclose(mase(y_true, y_pred, y_train, m=2), 1.0)


def test_registry_contains_all_models():
    from bottleneck_sim.forecasting import MODEL_REGISTRY
    for key in ["arima", "sarima", "hybrid_sarima_xgb", "ml_direct_xgb"]:
        assert key in MODEL_REGISTRY
