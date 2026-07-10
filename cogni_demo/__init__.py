"""Local-only graphical control plane for the Cogni-OS validation demo."""

from .protocol import EVENT_SENTINEL, ProtocolError, parse_event_line

__all__ = ["EVENT_SENTINEL", "ProtocolError", "parse_event_line"]
