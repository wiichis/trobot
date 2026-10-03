"""Alerta de ventana de compra / venta de BTC (03/10/2026)."""
import json

import numpy as np
import pandas as pd
import pytest

import pkg.btc_alertas as ba

T0 = pd.Timestamp("2026-10-03 00:02", tz="UTC")


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "OUT_DIR", tmp_path / "btc_bots")
    avisos = []
    monkeypatch.setattr(ba, "_emitir", lambda **kw: avisos.append(kw))
    return tmp_path, avisos


def velas_de(cierres, ahora):
    """Velas diarias al estilo BingX terminando en la vela EN FORMACIÓN de `ahora`."""
    fechas = pd.date_range(end=ahora.floor("D"), periods=len(cierres) + 1, freq="D")
    cierres = list(cierres) + [cierres[-1]]
    return [{"date": d, "open": c, "high": c, "low": c, "close": c} for d, c in zip(fechas, cierres)]


def fuente(cierres, ahora):
    return {"velas": lambda interval, limit: velas_de(cierres, ahora)}


SUBE = [100.0] * 150 + [100 * 1.01 ** k for k in range(1, 51)]          # sobre la media de 100
CAE = SUBE + [SUBE[-1] * 0.96 ** k for k in range(1, 13)]                # -38%: bajo la media y RSI < 30


def test_arranque_avisa_el_estado_actual(entorno):
    tmp, avisos = entorno
    r = ba.run_btc_alertas(ahora=T0, fuentes=fuente(SUBE, T0))
    assert r["ventana"].startswith("Estado actual: 🟢 COMPRA")
    assert len(avisos) == 1 and "media de 100 días" in avisos[0]["precio_referencia"]
    e = json.loads((tmp / "btc_bots/ventana_estado.json").read_text())
    assert e["tendencia"] == 1 and e["caida"] == 0
    assert e["barra"] == str(T0.floor("D") - pd.Timedelta(days=1))          # la vela cerrada


def test_misma_vela_no_repite_y_sin_cambio_no_avisa(entorno):
    _, avisos = entorno
    ba.run_btc_alertas(ahora=T0, fuentes=fuente(SUBE, T0))
    assert ba.run_btc_alertas(ahora=T0 + pd.Timedelta(hours=1), fuentes=fuente(SUBE, T0)) == {}
    t1 = T0 + pd.Timedelta(days=1)
    assert ba.run_btc_alertas(ahora=t1, fuentes=fuente(SUBE + [SUBE[-1] * 1.01], t1)) == {}
    assert len(avisos) == 1


def test_caida_abre_venta_y_compra_por_caida_en_un_solo_aviso(entorno):
    tmp, avisos = entorno
    ba.run_btc_alertas(ahora=T0, fuentes=fuente(SUBE, T0))
    t1 = T0 + pd.Timedelta(days=12)
    r = ba.run_btc_alertas(ahora=t1, fuentes=fuente(CAE, t1))
    assert r["ventana"].startswith("🔴 VENTA")
    assert r["caida_fuerte"].startswith("🟢 COMPRA por caída fuerte")
    assert len(avisos) == 2
    log = pd.read_csv(tmp / "btc_bots/ventana_log.csv")
    assert len(log) == 2 and "VENTA" in log.aviso.iat[-1]


def test_estado_coincide_con_las_reglas_de_los_bots(entorno):
    from pkg.btc_reglas import posiciones, rsi
    rng = np.random.default_rng(7)
    cierres = list(30_000 * np.cumprod(1 + rng.normal(0, 0.03, 400)))
    v = pd.DataFrame({"open": cierres, "high": cierres, "low": cierres, "close": cierres},
                     index=pd.date_range("2025-01-01", periods=400, freq="D", tz="UTC"))
    e = ba.evaluar(v)
    assert e["tendencia"] == posiciones(v, "T1", True).iat[-1]
    assert e["caida"] == posiciones(v, "R1", True).iat[-1]
    assert e["media100"] == pytest.approx(v.close.iloc[-100:].mean())
    assert e["rsi"] == pytest.approx(rsi(v.close).iat[-1])


def test_texto_para_el_resumen(entorno):
    ba.run_btc_alertas(ahora=T0, fuentes=fuente(SUBE, T0))
    t = ba.texto_estado(ba.leer_estado())
    assert t.startswith("COMPRA (tendencia alcista) desde") and "referencia media 100 d" in t


def test_nunca_levanta(entorno):
    tmp, avisos = entorno
    def explota(*a):
        raise RuntimeError("API caída")
    assert ba.run_btc_alertas(ahora=T0, fuentes={"velas": explota}) is None
    assert ba.run_btc_alertas(ahora=T0, fuentes=fuente([100.0] * 50, T0)) is None     # pocas velas
    assert avisos == [] and not (tmp / "btc_bots/ventana_estado.json").exists()
