from app.setup_lifecycle import is_allowed_transition
from app.state_store import SignalStore

import pytest


def test_state_machine_allows_only_forward_and_explicit_reset_edges():
    assert is_allowed_transition(None, "ARMED")
    assert is_allowed_transition("ARMED", "CONFIRMING")
    assert is_allowed_transition("CONFIRMING", "ARMED")
    assert is_allowed_transition("CONFIRMING", "TRIGGER_CONFIRMED")
    assert is_allowed_transition("TRIGGER_CONFIRMED", "ALERT_QUEUED")
    assert is_allowed_transition("ALERT_QUEUED", "TELEGRAM_SENT")
    assert is_allowed_transition("ARMED", "EXPIRED")
    assert is_allowed_transition("ARMED", "INVALIDATED")
    assert not is_allowed_transition("TELEGRAM_SENT", "ARMED")
    assert not is_allowed_transition("EXPIRED", "CONFIRMING")
    assert not is_allowed_transition("INVALIDATED", "TRIGGER_CONFIRMED")
    assert not is_allowed_transition("ARMED", "TELEGRAM_SENT")


def test_setup_lifecycle_transitions_are_durable_and_ordered(tmp_path):
    path = tmp_path / "state.sqlite3"
    store = SignalStore(str(path))
    candle_time = "2026-11-20T10:00:00+00:00"

    assert store.transition_setup("BTCUSD", "ORB", candle_time, None, "ARMED", "setup_armed")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "ARMED", "CONFIRMING", "quote_crossed")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "CONFIRMING", "ARMED", "quote_reversed")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "ARMED", "TRIGGER_CONFIRMED", "held_breakout")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "TRIGGER_CONFIRMED", "ALERT_QUEUED", "outbox_enqueued")

    reopened = SignalStore(str(path))
    state = reopened.setup_lifecycle("BTCUSD", "ORB", candle_time)
    assert state["state"] == "ALERT_QUEUED"
    assert state["version"] == 5
    events = reopened.setup_lifecycle_events("BTCUSD", "ORB", candle_time)
    assert [event["to_state"] for event in events] == [
        "ARMED", "CONFIRMING", "ARMED", "TRIGGER_CONFIRMED", "ALERT_QUEUED"
    ]
    assert events[2]["from_state"] == "CONFIRMING"
    assert events[2]["reason"] == "quote_reversed"


def test_setup_lifecycle_rejects_illegal_or_stale_transitions(tmp_path):
    store = SignalStore(str(tmp_path / "state.sqlite3"))
    candle_time = "2026-11-20T10:00:00+00:00"
    with pytest.raises(ValueError):
        store.transition_setup("BTCUSD", "MEX", candle_time, None, "CONFIRMING", "invalid_initial")
    assert store.transition_setup("BTCUSD", "MEX", candle_time, None, "ARMED", "setup_armed")
    assert store.transition_setup("BTCUSD", "MEX", candle_time, "ARMED", "CONFIRMING", "quote_crossed")
    assert not store.transition_setup("BTCUSD", "MEX", candle_time, "ARMED", "EXPIRED", "stale_worker")
    assert store.setup_lifecycle("BTCUSD", "MEX", candle_time)["state"] == "CONFIRMING"


def test_orb_and_mex_lifecycle_keys_are_independent(tmp_path):
    store = SignalStore(str(tmp_path / "state.sqlite3"))
    candle_time = "2026-11-20T10:00:00+00:00"
    assert store.transition_setup("BTCUSD", "ORB", candle_time, None, "ARMED", "orb")
    assert store.transition_setup("BTCUSD", "MEX", candle_time, None, "ARMED", "mex")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "ARMED", "EXPIRED", "orb_expired")
    assert store.setup_lifecycle("BTCUSD", "MEX", candle_time)["state"] == "ARMED"
    assert store.setup_lifecycle("BTCUSD", "ORB", candle_time)["state"] == "EXPIRED"



def test_setup_lifecycle_health_reports_active_and_terminal_states(tmp_path):
    store = SignalStore(str(tmp_path / "state.sqlite3"))
    candle_time = "2026-11-20T10:00:00+00:00"
    assert store.transition_setup("BTCUSD", "ORB", candle_time, None, "ARMED", "orb_armed")
    assert store.transition_setup("ETHUSD", "MEX", candle_time, None, "ARMED", "mex_armed")
    assert store.transition_setup("ETHUSD", "MEX", candle_time, "ARMED", "CONFIRMING", "mex_crossed")
    assert store.transition_setup("BTCUSD", "ORB", candle_time, "ARMED", "EXPIRED", "orb_expired")

    health = store.setup_lifecycle_health()
    assert health["total"] == 2
    assert health["active"] == 1
    assert health["terminal"] == 1
    assert health["states"] == {"CONFIRMING": 1, "EXPIRED": 1}
