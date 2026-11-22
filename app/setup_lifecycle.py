"""Validated lifecycle transitions shared by independent signal strategies."""

STATES = frozenset({
    "ARMED", "CONFIRMING", "TRIGGER_CONFIRMED", "ALERT_QUEUED",
    "TELEGRAM_SENT", "EXPIRED", "INVALIDATED",
})

_ALLOWED = {
    None: frozenset({"ARMED"}),
    "ARMED": frozenset({"CONFIRMING", "TRIGGER_CONFIRMED", "EXPIRED", "INVALIDATED"}),
    "CONFIRMING": frozenset({"ARMED", "TRIGGER_CONFIRMED", "EXPIRED", "INVALIDATED"}),
    "TRIGGER_CONFIRMED": frozenset({"ALERT_QUEUED"}),
    "ALERT_QUEUED": frozenset({"TELEGRAM_SENT"}),
    "TELEGRAM_SENT": frozenset(),
    "EXPIRED": frozenset(),
    "INVALIDATED": frozenset(),
}


def is_allowed_transition(from_state: str | None, to_state: str) -> bool:
    """Return whether a lifecycle edge is legal; terminal states never regress."""
    return to_state in STATES and to_state in _ALLOWED.get(from_state, frozenset())


def require_transition(from_state: str | None, to_state: str) -> None:
    if not is_allowed_transition(from_state, to_state):
        raise ValueError(f"Illegal setup lifecycle transition: {from_state} -> {to_state}")
