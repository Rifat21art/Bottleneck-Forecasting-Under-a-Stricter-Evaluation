import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from bottleneck_sim.utils.features import seasonal_lag, PERIODS_PER_WEEK
from bottleneck_sim.forecasting.classifier import BottleneckAlertClassifier
from bottleneck_sim.forecasting.hybrid_sarima_xgb import HybridSARIMAXGBForecaster
from bottleneck_sim.forecasting.ml_direct_model import MLDirectForecaster

N_TRAIN, N_TEST = 3 * PERIODS_PER_WEEK, 2 * 96


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-01", periods=N_TRAIN + N_TEST, freq="15min")
    daily = 0.3 + 0.2 * np.sin(2 * np.pi * np.arange(len(idx)) / 96)
    usage = pd.Series(100 * daily + rng.normal(0, 5, len(idx)), index=idx)
    util = pd.Series(np.clip(daily + rng.normal(0, 0.05, len(idx)), 0, 1), index=idx)
    return util, usage


def test_seasonal_lag_never_reads_past_origin():
    idx = pd.date_range("2018-01-01", periods=4 * PERIODS_PER_WEEK, freq="15min")
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)  # value == position
    origin = idx[2 * PERIODS_PER_WEEK - 1]
    out = seasonal_lag(s, idx, origin)
    src_pos = out.dropna().to_numpy()
    assert (src_pos <= 2 * PERIODS_PER_WEEK - 1).all()
    # training rows use plain lag-one-week
    assert out.iloc[PERIODS_PER_WEEK] == 0.0
    # 1..week steps ahead use last week; week+1.. steps ahead use two weeks back
    assert out.iloc[2 * PERIODS_PER_WEEK] == PERIODS_PER_WEEK
    assert out.iloc[3 * PERIODS_PER_WEEK] == PERIODS_PER_WEEK


@pytest.mark.parametrize("model_cls", [MLDirectForecaster, HybridSARIMAXGBForecaster])
def test_forecast_ignores_exogenous_values_after_origin(series, model_cls):
    util, usage = series
    train, test_idx = util.iloc[:N_TRAIN], util.index[N_TRAIN:]
    tampered = usage.copy()
    tampered.iloc[N_TRAIN:] = 1e6  # garbage "future" usage

    fc_a = model_cls().fit(train, usage_series=usage).predict(N_TEST, index=test_idx)
    fc_b = model_cls().fit(train, usage_series=tampered).predict(N_TEST, index=test_idx)
    pd.testing.assert_series_equal(fc_a, fc_b)
    assert len(fc_a) == N_TEST and fc_a.between(0, 1).all()


def test_classifier_labels_and_split(series):
    util, usage = series
    train_idx, test_idx = util.index[:N_TRAIN], util.index[N_TRAIN:]
    clf = BottleneckAlertClassifier().fit(util, usage, train_index=train_idx)
    labels = clf.true_labels(test_idx)
    # last timestamp has no next value, so it has no label (rather than a fake "0")
    assert util.index[-1] not in labels.index
    expected = (util.shift(-1) > clf.threshold_).astype(int).loc[labels.index]
    pd.testing.assert_series_equal(labels, expected, check_names=False)
    assert "util_lag0" in clf.feature_importances().index
    assert len(clf.predict(test_idx)) == len(labels)
