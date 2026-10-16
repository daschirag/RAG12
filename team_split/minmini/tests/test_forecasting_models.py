"""A3 models: train-only imputation, same scored rows, causal windows, determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backend.forecasting.features import build_features, split_by_target_day
from src.backend.forecasting.metrics import mae, mape, r2, rmse
from src.backend.forecasting.models import (
    PLANT_CAPACITY_KW,
    DailyPersistence,
    LSTMForecaster,
    Persistence,
    RandomForest,
    SequenceEncoder,
    TCNForecaster,
    TrainConfig,
    TrainOnlyImputer,
    XGBoost,
    eval_mask,
    night_mask,
    postprocess,
)
from src.backend.forecasting.preprocess import TARGET, load_plant_raw, preprocess

FAST = TrainConfig(max_epochs=3, patience=2, hidden=8)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    gen, weather = load_plant_raw(1)
    return preprocess(gen, weather, plant=1).frame


@pytest.fixture(scope="module")
def parts(frame):
    return split_by_target_day(build_features(frame))


# --------------------------------------------------------------- metrics

def test_metrics_known_values():
    y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 8.0])
    assert mae(y, p) == 1.0
    assert rmse(y, p) == 2.0
    assert r2(y, y) == 1.0
    assert r2(y, np.full(4, y.mean())) == 0.0
    assert mape(y, p, min_true=4.0) == 100.0      # only the last row qualifies
    with pytest.raises(ValueError):
        mae(y, p[:3])
    with pytest.raises(ValueError):
        mae(y, np.array([1.0, np.nan, 3.0, 4.0]))


# ------------------------------------------------------------- imputation

def test_imputer_uses_training_rows_only(parts):
    train, val = parts["train"], parts["val"]
    imp = TrainOnlyImputer().fit(train.X)
    before = imp.by_group_.copy()
    # Fitting must not look at val: refit on train with val values altered -> same stats.
    val_poisoned = val.X.copy() * 1000
    assert imp.transform(val_poisoned).notna().all().all()
    pd.testing.assert_frame_equal(TrainOnlyImputer().fit(train.X).by_group_, before)
    # A missing value is replaced by the training median of its hour.
    row = val.X.iloc[[0]].copy()
    row["ac_roll_mean_16"] = np.nan
    filled = imp.transform(row)["ac_roll_mean_16"].iloc[0]
    assert filled == before.loc[row["hour"].iloc[0], "ac_roll_mean_16"]


def test_imputer_never_drops_rows(parts):
    test = parts["test"]
    out = TrainOnlyImputer().fit(parts["train"].X).transform(test.X)
    assert len(out) == len(test.X) and out.notna().all().all()


def test_transform_before_fit_raises(parts):
    with pytest.raises(RuntimeError):
        TrainOnlyImputer().transform(parts["val"].X)


# -------------------------------------------------------- post-processing

def test_postprocess_clips_and_zeros_night(parts):
    test = parts["test"]
    raw = np.linspace(-500, 2 * PLANT_CAPACITY_KW, len(test.y))
    out = postprocess(raw, test)
    assert out.min() >= 0 and out.max() <= PLANT_CAPACITY_KW
    assert (out[night_mask(test)] == 0).all()
    # Strict default rule: only slots with no irradiation on any previous day are zeroed,
    # a subset of the A2 night flag (envelope <= 0.05).
    assert (night_mask(test) <= ~test.meta["target_daytime"].to_numpy()).all()
    assert night_mask(test).sum() > 0


def test_eval_mask_excludes_imputed_targets(parts):
    test = parts["test"]
    m = eval_mask(test)
    assert m.sum() == (~test.meta["target_imputed"]).sum()


# --------------------------------------------------------------- baselines

def test_persistence_is_value_at_origin(parts, frame):
    train, val, test = parts["train"], parts["val"], parts["test"]
    model = Persistence().fit(train, val, frame)
    pred = model.predict(test, frame)
    observed = test.X["ac_now"].notna().to_numpy()
    np.testing.assert_array_equal(pred[observed], test.X["ac_now"].to_numpy()[observed])
    assert np.isfinite(pred).all()


def test_daily_persistence_uses_yesterday(parts, frame):
    test = parts["test"]
    pred = DailyPersistence().fit(parts["train"], parts["val"], frame).predict(test, frame)
    source = test.X.index + pd.Timedelta("1h") - pd.Timedelta("1D")
    expected = frame[TARGET].reindex(source).to_numpy()
    observed = ~np.isnan(expected)
    assert observed.sum() > 400                    # the check covers most test rows
    np.testing.assert_array_equal(pred[observed], expected[observed])


# ------------------------------------------------------------- tree models

@pytest.mark.parametrize("cls", [RandomForest, XGBoost])
def test_tree_models_are_seed_deterministic(parts, frame, cls, monkeypatch):
    monkeypatch.setattr(cls, "grid", {k: v[:1] for k, v in cls.grid.items()})
    a = cls(seed=0).fit(parts["train"], parts["val"], frame).predict(parts["test"], frame)
    b = cls(seed=0).fit(parts["train"], parts["val"], frame).predict(parts["test"], frame)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(parts["test"].y),) and np.isfinite(a).all()


def test_tree_model_feature_subset(parts, frame, monkeypatch):
    monkeypatch.setattr(XGBoost, "grid", {"max_depth": [3], "learning_rate": [0.1]})
    cols = [c for c in parts["train"].feature_names if c not in ("day", "month")]
    model = XGBoost(seed=0, features=cols).fit(parts["train"], parts["val"], frame)
    assert list(model.model_.feature_names_in_) == cols


def test_tree_grid_selects_on_validation(parts, frame, monkeypatch):
    monkeypatch.setattr(XGBoost, "grid", {"max_depth": [3], "learning_rate": [0.03, 0.1]})
    model = XGBoost(seed=0).fit(parts["train"], parts["val"], frame)
    assert model.params_["learning_rate"] in (0.03, 0.1)
    assert model.val_mae_ > 0


# ---------------------------------------------------------- sequence models

def test_sequence_window_ends_at_origin_and_is_causal(parts, frame):
    train, test = parts["train"], parts["test"]
    enc = SequenceEncoder(lookback=24).fit(train, frame)
    seq, ctx = enc.transform(test, frame)
    assert seq.shape == (len(test.y), 24, enc.n_inputs) and ctx.shape[0] == len(test.y)
    # Last step of the window is the value at the origin t (scaled).
    t = test.X.index[100]
    i = test.X.index.get_loc(t)
    ac_t = frame.loc[t, TARGET]
    if not np.isnan(ac_t):
        assert seq[i, -1, 0] == pytest.approx((ac_t - enc.seq_mean_[0]) / enc.seq_std_[0], rel=1e-5)
    # Corrupting everything after t leaves the window for t unchanged.
    dirty = frame.copy()
    after = dirty.index > t
    dirty.loc[after, [TARGET, "irradiation", "ambient_temp", "module_temp"]] = 9999.0
    seq_dirty, _ = enc.transform(test, dirty)
    np.testing.assert_array_equal(seq[i], seq_dirty[i])


def test_sequence_encoder_statistics_from_training_period(parts, frame):
    train = parts["train"]
    enc = SequenceEncoder().fit(train, frame)
    dirty = frame.copy()
    dirty.loc[dirty.index > train.X.index.max(), TARGET] = 1e7
    enc2 = SequenceEncoder().fit(train, dirty)
    np.testing.assert_array_equal(enc.seq_mean_, enc2.seq_mean_)
    np.testing.assert_array_equal(enc.seq_std_, enc2.seq_std_)


@pytest.mark.parametrize("cls", [LSTMForecaster, TCNForecaster])
def test_sequence_models_deterministic_and_finite(parts, frame, cls):
    a = cls(seed=0, train_cfg=FAST).fit(parts["train"], parts["val"], frame).predict(parts["test"], frame)
    b = cls(seed=0, train_cfg=FAST).fit(parts["train"], parts["val"], frame).predict(parts["test"], frame)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (len(parts["test"].y),) and np.isfinite(a).all()


def test_tcn_is_causal_inside_the_window():
    """Changing a later step of the input window must not change earlier conv outputs."""
    import torch

    from src.backend.forecasting.models import _TCNNet

    torch.manual_seed(0)
    net = _TCNNet.build(n_inputs=3, n_ctx=1, hidden=4)
    x = torch.randn(1, 3, 20)
    y1 = net.tcn(x)
    x2 = x.clone()
    x2[:, :, 15:] += 10.0
    y2 = net.tcn(x2)
    assert torch.equal(y1[:, :, :15], y2[:, :, :15])
    assert not torch.equal(y1[:, :, 15:], y2[:, :, 15:])
