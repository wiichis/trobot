from __future__ import annotations


def test_lifecycle_emit_uses_dispatcher_once(monkeypatch):
    import pkg.lifecycle_events as le

    calls = []

    class _FakeDispatcher:
        def emit(self, *, category, severity="INFO", force=False, **fields):
            calls.append(
                {
                    "category": category,
                    "severity": severity,
                    "force": force,
                    "fields": dict(fields),
                }
            )
            return {"sent": True, "detail": "mocked", "ts_utc": "2026-01-01T00:00:00Z"}

    monkeypatch.setattr(le, "get_lifecycle_dispatcher", lambda: _FakeDispatcher())
    out = le.emit_lifecycle_event("tp2_submitted", "INFO", symbol="HBAR-USDT", qty=1.23)

    assert out["sent"] is True
    assert len(calls) == 1
    assert calls[0]["category"] == "tp2_submitted"
    assert calls[0]["fields"]["symbol"] == "HBAR-USDT"
