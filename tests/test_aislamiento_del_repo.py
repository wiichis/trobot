from __future__ import annotations

import pandas as pd

from conftest import ARCHIVOS_DEL_REPO


def test_lifecycle_y_ledger_escriben_en_tmp_y_no_mandan_telegram(tmp_path, monkeypatch):
    """La config de Telegram trackeada tiene `enabled: true`: con credenciales en el
    entorno, un test que emitía un evento mandaba una alerta real."""
    import pkg.execution_ledger as el
    import pkg.lifecycle_events as le
    import pkg.telegram_alerts as ta

    monkeypatch.setenv("TROBOT_TELEGRAM_BOT_TOKEN", "token_test")
    monkeypatch.setenv("TROBOT_TELEGRAM_CHAT_ID", "123456")
    posts = []
    monkeypatch.setattr(ta.requests, "post", lambda *a, **k: posts.append((a, k)))

    out = le.emit_lifecycle_event("tick_size_implausible", "WARN", symbol="ONDO-USDT", tick=0.01)
    el.append_execution_ledger_event("tp1_filled", symbol="ONDO-USDT")

    assert posts == []
    assert out["detail"] == "telegram_disabled"
    for path in (le.LIFECYCLE_LOG_CSV, el.DEFAULT_EXECUTION_LEDGER_PATH):
        assert path.parent == tmp_path / "archivos"
        assert ARCHIVOS_DEL_REPO not in path.parents
        assert len(pd.read_csv(path)) == 1


def test_pull_de_velas_escribe_el_agregado_30m_en_tmp(tmp_path, monkeypatch):
    import pkg.price_bingx_5m as px

    ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=10)).floor("5min")
    vela = {"symbol": "ETH-USDT", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10, "date": ts}
    monkeypatch.setattr(px, "currencies_list", lambda: ["ETH-USDT"])
    monkeypatch.setattr(px, "_fetch_bingx_candles", lambda *_a, **_k: [vela])

    px.price_bingx_5m()

    assert px.CSV_30M_PATH == tmp_path / "archivos" / "cripto_price_30m.csv"
    assert len(pd.read_csv(px.CSV_30M_PATH)) == 1
