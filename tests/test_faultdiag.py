"""B4: SCADA loading safeguards, leave-one-asset-out protocol, metrics and attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backend.faultdiag.data import EventData, base_sensor, load_event, robust_scale, sensor_columns
from src.backend.faultdiag.models import (
    attribute_components,
    classification_metrics,
    event_table,
    isolation_forest,
    leave_one_asset_out,
)


@pytest.fixture(scope="module")
def event73() -> EventData:
    return load_event("A", 73)


def test_status_type_never_used(event73):
    assert not any("status" in c for c in event73.X.columns)
    assert not any("status" in c for c in event73.train_X.columns)


def test_base_sensor_and_angle_columns():
    assert base_sensor("sensor_18_std") == "sensor_18"
    assert base_sensor("sin_sensor_1_avg") == "sensor_1"
    assert base_sensor("sensor_44") == "sensor_44"
    sensors, angles = sensor_columns(["time_stamp", "status_type_id", "sensor_1_avg", "sensor_5_avg", "sensor_5_max",
                                      "sensor_0_avg"])
    assert angles == ["sensor_1_avg", "sensor_5_avg"]
    assert sensors == ["sensor_5_max", "sensor_0_avg"]          # only the average of an angle sensor is an angle


def test_robust_scale_uses_train_rows_only(event73):
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 100], "b": [5.0] * 5})
    s = robust_scale(df.iloc[:4])
    assert s.loc["a", "median"] == 2.5 and s.loc["a", "iqr"] == pytest.approx(1.5)
    assert s.loc["b", "iqr"] == 1.0                               # constant column -> scale 1
    # The real event: window values do not influence the scaling statistics.
    assert event73.scale.shape[0] == event73.X.shape[1]


def test_window_rows_and_labels(event73):
    assert len(event73.X) == len(event73.y) and set(event73.y) == {1}
    assert event73.X.notna().all().all() and np.isfinite(event73.X.to_numpy()).all()


def _toy_events(n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    events = []
    for a in range(n_assets):
        for label in (0, 1):
            X = pd.DataFrame(rng.normal(size=(40, 4)) + (2.0 if label else 0.0), columns=list("abcd"))
            events.append(EventData("A", a * 10 + label, a, "anomaly" if label else "normal", X,
                                    np.full(40, label), pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd")),
                                    pd.DataFrame()))
    return events


@pytest.mark.parametrize("model", ["logistic_regression", "random_forest", "xgboost"])
def test_leave_one_asset_out_never_trains_on_the_test_asset(model, monkeypatch):
    import src.backend.faultdiag.models as m

    events = _toy_events()
    seen = []
    original = m._matrix

    def spy(evts):
        seen.append({e.asset_id for e in evts})
        return original(evts)

    monkeypatch.setattr(m, "_matrix", spy)
    rows = leave_one_asset_out(events, model)
    assert seen == [{1, 2}, {0, 2}, {0, 1}]                      # train assets when asset 0, 1, 2 is held out
    assert len(rows) == sum(len(e.y) for e in events)
    assert classification_metrics(rows.y, rows.flag, rows.score)["auc"] > 0.95   # separable toy data


def test_supervised_and_isolation_forest_are_deterministic():
    events = _toy_events()
    a = leave_one_asset_out(events, "random_forest", seed=0)
    b = leave_one_asset_out(events, "random_forest", seed=0)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(isolation_forest(events, 0), isolation_forest(events, 0))


def test_event_table_rules():
    rows = pd.DataFrame({"event_id": [1, 1, 2, 2], "asset_id": [0, 0, 1, 1], "y": [1, 1, 0, 0],
                         "score": [0.9, 0.4, 0.2, 0.1], "flag": [1, 0, 0, 0]})
    sup = event_table(rows, "xgboost")
    assert sup.loc[sup.event_id == 1, "event_prob"].item() == pytest.approx(0.65)
    iso = event_table(rows, "isolation_forest")
    assert iso.loc[iso.event_id == 1, "event_prob"].item() == 0.5 and iso.loc[iso.event_id == 1, "event_flag"].item() == 1


def test_metrics_known_values():
    m = classification_metrics([1, 1, 0, 0], [1, 0, 0, 1], [0.9, 0.4, 0.2, 0.6])
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["auc"] == pytest.approx(0.75)
    assert np.isnan(classification_metrics([1, 1], [1, 1], [0.5, 0.6])["auc"])


def test_select_threshold_maximises_f1_and_prefers_fewer_alarms():
    from src.backend.faultdiag.models import select_threshold

    from sklearn.metrics import f1_score

    t = select_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert 0.2 <= t < 0.8                                            # perfect separation
    # Property check on random cases (many with ties): the chosen threshold has the maximum F1 over the
    # candidate set and is the highest threshold among those that tie for it.
    rng = np.random.default_rng(0)
    ties_seen = 0
    for _ in range(200):
        s = np.round(rng.random(8), 1)
        y = rng.integers(0, 2, 8)
        chosen = select_threshold(s, y)
        u = np.unique(s)
        cands = np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1]]])
        f1 = np.array([f1_score(y, (s > c).astype(int), zero_division=0) for c in cands])
        best = cands[f1 == f1.max()]
        ties_seen += len(best) > 1
        assert chosen == pytest.approx(best.max())
    assert ties_seen > 0


def test_nested_threshold_never_uses_the_test_asset():
    from src.backend.faultdiag.models import nested_threshold_eval

    events = pd.DataFrame({"asset_id": [0, 0, 1, 1, 2, 2], "y": [0, 1, 0, 1, 0, 1],
                           "event_prob": [0.1, 0.9, 0.2, 0.8, 0.95, 0.05]})     # asset 2 is inverted
    res = nested_threshold_eval(events)
    # Asset 2's threshold is chosen on assets 0 and 1 only, so its inverted scores are misclassified.
    a2 = res[res.asset_id == 2]
    assert (a2.event_flag_nested.to_numpy() == [1, 0]).all()
    assert res.groupby("asset_id")["threshold"].nunique().eq(1).all()


def test_attribution_ranks_the_deviating_component(event73):
    shifted = event73.X.copy()
    hydraulic_cols = [c for c in shifted.columns if base_sensor(c) == "sensor_41"]   # oil temperature in hydraulic group
    assert hydraulic_cols
    shifted[hydraulic_cols] = 30.0
    e = EventData("A", 73, event73.asset_id, "anomaly", shifted, event73.y, event73.train_X, event73.scale)
    assert attribute_components(e).index[0] == "hydraulic_system"
