import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def smape(y_true, y_pred):
    """Symmetric MAPE (%). Unstable when many values are near zero - prefer MAE/RMSE/MASE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def mase(y_true, y_pred, y_train, m=96):
    """
    Mean absolute scaled error (Hyndman & Koehler, 2006): test MAE divided by the
    in-sample MAE of a seasonal-naive forecast with period `m`. Values < 1 beat
    seasonal-naive; well-defined when the series contains zeros.
    """
    y_true, y_pred, y_train = map(np.asarray, (y_true, y_pred, y_train))
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y_true - y_pred)) / scale) if scale > 0 else float("nan")


def evaluate(y_true, y_pred, y_train=None, m=96):
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, m=m)
    return out
