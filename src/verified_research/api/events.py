"""Execution event emitter context and integration hooks for live streaming."""

from contextvars import ContextVar
from typing import Any, Callable

# Callback signature: (event_type: str, node: str | None, data: dict[str, Any]) -> None
EventEmitterType = Callable[[str, str | None, dict[str, Any]], None]

_current_event_emitter: ContextVar[EventEmitterType | None] = ContextVar(
    "current_event_emitter", default=None
)


def set_execution_event_emitter(emitter: EventEmitterType | None) -> None:
    """Set the active event emitter for the current execution context."""
    _current_event_emitter.set(emitter)


def get_execution_event_emitter() -> EventEmitterType | None:
    """Retrieve the active event emitter, if configured."""
    return _current_event_emitter.get()


def emit_live_event(event_type: str, node: str | None = None, data: dict[str, Any] | None = None) -> None:
    """Convenience function to emit a live execution event if an emitter is active."""
    emitter = _current_event_emitter.get()
    if emitter is not None:
        try:
            emitter(event_type, node, data or {})
        except Exception:
            pass
