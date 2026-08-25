from __future__ import annotations

import pandas as pd
import pytest


def _candle(sym, ts, o, h, l, c, v=100.0):
    return {"symbol": sym, "open": o, "high": h, "low": l, "close": c,
            "volume": v, "date": ts}


def _vela_reciente(minutos_atras: int = 10) -> pd.Timestamp:
    """Timestamp de vela 5m alineado, dentro de la ventana de retención.

    `price_bingx_5m` purga lo anterior a SIGNAL_HISTORY_DAYS (30 d). Con fechas fijas
    estos tests caducaban en silencio: pasaban hasta que el calendario dejaba atrás la
    fecha del fixture y después fallaban por CSV vacío, sin que hubiera regresión.
    """
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=minutos_atras)).floor("5min")


def test_la_vela_en_formacion_se_corrige_al_cerrar(tmp_path, monkeypatch):
    """El pull corre en :01,:06,… y guarda la vela de 5m aún abierta. Con el filtro
    `>` nunca se volvía a bajar: quedaba con high/low truncados y close a mitad de
    vela, y TODOS los indicadores del live se calculaban sobre esos closes."""
    import pkg.price_bingx_5m as px

    csv = tmp_path / "cripto_price_5m.csv"
    monkeypatch.setattr(px, "CSV_PATH", csv)
    monkeypatch.setattr(px, "currencies_list", lambda: ["ETH-USDT"])

    t0 = _vela_reciente(10)
    t1 = t0 + pd.Timedelta(minutes=5)

    # Estado previo: la vela t0 se guardó en formación (rango parcial).
    pd.DataFrame([_candle("ETH-USDT", t0, 1933.95, 1935.63, 1933.31, 1935.63)]).to_csv(csv, index=False)

    # El API ya devuelve t0 CERRADA (rango real, mayor) y t1 en formación.
    cerrada = _candle("ETH-USDT", t0, 1933.95, 1935.82, 1930.26, 1930.29)
    nueva = _candle("ETH-USDT", t1, 1930.26, 1931.00, 1930.10, 1930.80)
    monkeypatch.setattr(px, "_fetch_bingx_candles", lambda _s, _n, **_k: [cerrada, nueva])

    px.price_bingx_5m()

    out = pd.read_csv(csv)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    fila = out[out["date"] == t0].iloc[0]

    assert fila["close"] == pytest.approx(1930.29), "la vela cerrada debe reemplazar a la parcial"
    assert fila["high"] == pytest.approx(1935.82)
    assert fila["low"] == pytest.approx(1930.26)
    assert len(out[out["date"] == t0]) == 1, "no debe duplicarse la vela"
    assert t1 in set(out["date"]), "la vela nueva debe entrar igual"


def test_no_se_pierden_velas_si_el_ciclo_se_atrasa(tmp_path, monkeypatch):
    """fetch_limit=3 da margen cuando el scheduler se salta un ciclo."""
    import pkg.price_bingx_5m as px

    csv = tmp_path / "cripto_price_5m.csv"
    monkeypatch.setattr(px, "CSV_PATH", csv)
    monkeypatch.setattr(px, "currencies_list", lambda: ["ETH-USDT"])

    base = _vela_reciente(15)
    ts = [base + pd.Timedelta(minutes=5 * i) for i in range(3)]
    pd.DataFrame([_candle("ETH-USDT", ts[0], 10, 10, 10, 10)]).to_csv(csv, index=False)

    limites = {}

    def _fetch(_s, n, **_k):
        limites["n"] = n
        return [_candle("ETH-USDT", t, 20, 21, 19, 20) for t in ts]

    monkeypatch.setattr(px, "_fetch_bingx_candles", _fetch)
    px.price_bingx_5m()

    out = pd.read_csv(csv)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    assert limites["n"] == 3
    assert len(out) == 3, "deben quedar las 3 velas sin huecos"
    assert out[out["date"] == ts[0]].iloc[0]["close"] == pytest.approx(20)
