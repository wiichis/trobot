from __future__ import annotations

import pandas as pd
import pytest


def _bars(offsets_min, **cols):
    """Barras de 5m cuyo timestamp es `now - offset`."""
    now = pd.Timestamp.now(tz="UTC").floor("min")
    df = pd.DataFrame({"date": [now - pd.Timedelta(minutes=m) for m in offsets_min]})
    for k, v in cols.items():
        df[k] = v
    return df.sort_values("date").reset_index(drop=True)


def test_devuelve_la_penultima_cuando_la_ultima_esta_en_formacion():
    """La última fila del CSV es la vela en formación: volumen parcial -> VOL_OK
    siempre False. Decidir sobre ella vetaba toda entrada (0 órdenes en 3 días)."""
    from pkg.indicadores import last_closed_bar

    # barra de hace 2 min = en formación (cierra en 3 min); la de hace 7 min ya cerró
    df = _bars([7, 2], tag=["cerrada", "en_formacion"])
    row = last_closed_bar(df)

    assert row is not None
    assert row["tag"] == "cerrada"


def test_devuelve_la_ultima_si_ya_cerro():
    from pkg.indicadores import last_closed_bar

    df = _bars([12, 7], tag=["vieja", "cerrada"])
    row = last_closed_bar(df)

    assert row["tag"] == "cerrada"


def test_none_si_ninguna_barra_cerro_todavia():
    from pkg.indicadores import last_closed_bar

    df = _bars([1], tag=["en_formacion"])
    assert last_closed_bar(df) is None


def test_none_con_dataframe_vacio():
    from pkg.indicadores import last_closed_bar

    assert last_closed_bar(pd.DataFrame(columns=["date"])) is None
    assert last_closed_bar(None) is None


def test_ema_alert_ignora_la_señal_de_la_vela_en_formacion(tmp_path, monkeypatch):
    """Regresión 31/07: la señal de la vela en formación no debe disparar entrada;
    la de la última vela cerrada sí."""
    import pkg.indicadores as ind

    now = pd.Timestamp.now(tz="UTC").floor("min")
    df = pd.DataFrame({
        "symbol": ["AVAX-USDT", "AVAX-USDT"],
        "date": [now - pd.Timedelta(minutes=7), now - pd.Timedelta(minutes=2)],
        "close": [6.0, 6.5],
        "Long_Signal": [False, True],    # señal SOLO en la vela en formación
        "Short_Signal": [False, False],
    })
    path = tmp_path / "indicadores.csv"
    df.to_csv(path, index=False)
    monkeypatch.setattr(ind, "IND_CSV", str(path))
    monkeypatch.setattr(ind, "_read", lambda _p, _m=None: pd.read_csv(path, parse_dates=["date"]))

    assert ind.ema_alert("AVAX-USDT") == (None, None)

    # Ahora la señal está en la vela cerrada: debe dispararse.
    df.loc[0, "Long_Signal"] = True
    df.loc[1, "Long_Signal"] = False
    df.to_csv(path, index=False)

    price, tipo = ind.ema_alert("AVAX-USDT")
    assert tipo == "Alerta de LONG"
    assert price == pytest.approx(6.0)
