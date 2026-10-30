"""v2.0.0 - Bottleneck-alert classifier: predicts high-utilization risk `lead` periods ahead.

A rolling nowcast: the row for time t uses only information observed up to and
including t (current and lagged utilization, current energy usage, calendar)
to predict whether utilization at t+lead exceeds the alert threshold.

v3.1.0: configurable lead time, probability outputs, and an operating cut-off
chosen on a calibration period to respect a false-alarm budget
(`choose_cutoff`).
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..utils.features import make_calendar, PERIODS_PER_DAY


class BottleneckAlertClassifier:
    def __init__(self, threshold_quantile=0.80, lead=1, lags=(0, 1, 2, 4, 96), xgb_kwargs=None):
        self.threshold_quantile = threshold_quantile
        self.lead = lead
        self.lags = lags
        self.xgb_kwargs = xgb_kwargs or dict(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss",
        )
        self.model_ = None
        self.threshold_ = None
        self.cutoff_ = 0.5

    def _features(self, idx, usage_series, target_series):
        feat = make_calendar(idx)
        feat["usage_kwh"] = usage_series.reindex(idx).to_numpy()
        for lag in self.lags:
            feat[f"util_lag{lag}"] = target_series.shift(lag).reindex(idx)
        return feat

    def fit(self, series, usage_series, train_index):
        train_index = pd.DatetimeIndex(train_index)
        self.threshold_ = series.loc[train_index].quantile(self.threshold_quantile)
        feat = self._features(series.index, usage_series, series)
        ahead = series.shift(-self.lead)
        target = pd.Series(np.where(ahead.isna(), np.nan, (ahead > self.threshold_).astype(float)),
                           index=series.index, name="target")
        data = feat.join(target).dropna()

        # a row's label lies `lead` periods later, so the last `lead` training rows
        # (whose labels fall after the training period) are excluded
        train_rows = train_index[:-self.lead]
        train_data = data.loc[data.index.intersection(train_rows)]
        X, y = train_data.drop(columns=["target"]), train_data["target"].astype(int)
        pos_frac = y.mean()
        spw = (1 - pos_frac) / max(pos_frac, 1e-6)
        self.model_ = XGBClassifier(**self.xgb_kwargs, scale_pos_weight=spw)
        self.model_.fit(X, y)
        self._full_data = data
        return self

    def _rows(self, index):
        return self._full_data.loc[self._full_data.index.intersection(index)]

    def predict_proba(self, index):
        X = self._rows(index).drop(columns=["target"])
        return pd.Series(self.model_.predict_proba(X)[:, 1], index=X.index)

    def predict(self, index):
        return (self.predict_proba(index) >= self.cutoff_).astype(int)

    def true_labels(self, index):
        return self._rows(index)["target"].astype(int)

    def current_state(self, index):
        """Persistence baseline input: is utilization above the threshold right now?"""
        return (self._rows(index)["util_lag0"] > self.threshold_).astype(int)

    def feature_importances(self):
        cols = self._full_data.columns.drop("target")
        return pd.Series(self.model_.feature_importances_, index=cols).sort_values(ascending=False)


def choose_cutoff(proba, y_true, max_false_alarms_per_day, periods_per_day=PERIODS_PER_DAY):
    """
    Probability cut-off that maximizes recall subject to at most
    `max_false_alarms_per_day` false-alarm periods per day (ties -> fewer false alarms).
    """
    proba, y_true = np.asarray(proba, float), np.asarray(y_true, int)
    days = len(proba) / periods_per_day
    budget = max_false_alarms_per_day * days
    cut_values = np.unique(proba)[::-1]  # descending candidate cut-offs (alert if proba >= c)
    order = np.argsort(-proba, kind="stable")
    p_sorted, y_sorted = proba[order], y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    # number of alerts at cut-off c is the count of proba >= c
    n_alerts = len(proba) - np.searchsorted(np.sort(proba), cut_values, side="left")
    tp_c, fp_c = tp[n_alerts - 1], fp[n_alerts - 1]
    feasible = fp_c <= budget
    if not feasible.any():
        return float(np.nextafter(proba.max(), np.inf))  # no alerts at all
    best = np.flatnonzero(feasible)[np.argmax(tp_c[feasible])]  # first max -> highest cut-off, fewest FAs
    return float(cut_values[best])


def alert_metrics(y_true, y_pred, periods_per_day=PERIODS_PER_DAY):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "false_alarms_per_day": fp / (len(y_true) / periods_per_day),
            "confusion_matrix": [[tn, fp], [fn, tp]]}
