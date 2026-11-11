"""Compatibility helpers for actor attribution in the shared audit history."""

import json


_NOTE_PREFIX = "tla-audit-v1:"


def pack_actor_note(actor, note):
    return _NOTE_PREFIX + json.dumps(
        {"actor": actor or "Unknown", "note": note or ""},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def unpack_actor_note(note, actor=None):
    """Return actor/note, reading packed records used before the SQL migration."""
    if isinstance(note, str) and note.startswith(_NOTE_PREFIX):
        try:
            payload = json.loads(note[len(_NOTE_PREFIX):])
            return actor or payload.get("actor") or "Unknown", payload.get("note") or ""
        except (TypeError, ValueError):
            pass
    return actor or "Legacy", note or ""

