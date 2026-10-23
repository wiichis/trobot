"""Bot BTC por rangos en papel (22/10/2026): paridad con la simulación y bordes de persistencia."""
import json

import numpy as np
import pandas as pd
import pytest

import pkg.btc_rangos as br
from pkg import btc_rangos_motor as mm
from pkg import btc_rangos_reglas as rr

CFG = json.loads(br.CONFIG_PATH.read_text(encoding="utf-8"))


def serie(dias=75, semilla=7):
    idx = pd.date_range("2024-01-01", periods=dias * 24, freq="h")
    rng = np.random.default_rng(semilla)
    c = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, len(idx)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, len(idx)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    cfg = dict(CFG, enabled=True, modo="papel")
    f = tmp_path / "btc_rangos.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(br, "CONFIG_PATH", f)
    monkeypatch.setattr(br, "OUT_DIR", tmp_path / "papel")
    monkeypatch.setattr(br, "_AVISADO", set())
    avisos = []
    monkeypatch.setattr(br, "_emitir", lambda cat, sev, **kw: avisos.append((cat, sev, kw)))
    return {"cfg": cfg, "cfg_path": f, "avisos": avisos, "dir": tmp_path / "papel"}


def fuente(full, ahora):
    # devuelve también la vela en formación, como el API real
    return {"velas": lambda limit: full[full.index <= ahora.tz_localize(None)].tail(limit)}


def test_paridad_con_la_simulacion_corriendo_hora_a_hora_con_huecos(entorno):
    full = serie()
    inicio = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    assert br.run_btc_rangos(inicio, fuente(full, inicio))["nuevo"]
    t = inicio
    fin = pd.Timestamp("2024-03-15 00:30", tz="UTC")
    huecos = {pd.Timestamp("2024-02-20 05:30", tz="UTC"), pd.Timestamp("2024-03-03 01:30", tz="UTC")}
    while t < fin:
        t += pd.Timedelta(hours=1)
        if t in huecos:            # el server no corrió unas horas
            t += pd.Timedelta(hours=6)
        br.run_btc_rangos(t, fuente(full, t))
    # simulación de referencia sobre la serie completa
    datos = br.preparar(full, entorno["cfg"])
    ind_atr = rr.atr(full)
    hasta = pd.Timestamp("2024-03-15 00:00")  # última vela cerrada en la última corrida: 23:00
    r = mm.simular(full, datos["senal"], ind_atr, br.config_motor(entorno["cfg"]),
                   lambda k: float(entorno["cfg"]["dca_mensual_usdt"]), pd.Timestamp("2024-02-10 00:00"), hasta,
                   rango=datos["rango"])
    sim = pd.DataFrame(r.motor.operaciones)
    papel = pd.read_csv(entorno["dir"] / "operaciones.csv", parse_dates=["entrada_t", "salida_t"])
    assert len(sim) > 5, "la serie sintética debe generar operaciones"
    assert len(papel) == len(sim)
    assert (papel["slot"].to_numpy() == sim["slot"].to_numpy()).all()
    assert (papel["entrada_t"].to_numpy() == sim["entrada_t"].to_numpy()).all()
    assert np.allclose(papel["ganancia"], sim["ganancia"])
    estado = json.loads((entorno["dir"] / "estado.json").read_text())
    assert estado["motor"]["holding_btc"] == pytest.approx(r.motor.holding_btc)
    assert estado["motor"]["n_cierres"] == r.motor.n_cierres == 1      # cruzó el 03/03
    mensual = pd.read_csv(entorno["dir"] / "mensual.csv")
    assert len(mensual) == 1 and mensual["dca"].iloc[0] == 400
    cats = [a[0] for a in entorno["avisos"]]
    assert "btc_rangos_operaciones" in cats and "btc_rangos_cierre_mensual" in cats
    assert "btc_rangos_resumen" in cats


def test_estado_ilegible_no_escribe_y_avisa(entorno):
    full = serie()
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(full, t))
    f = entorno["dir"] / "estado.json"
    f.write_text("{roto", encoding="utf-8")
    t2 = t + pd.Timedelta(hours=1)
    assert br.run_btc_rangos(t2, fuente(full, t2)) is None
    assert f.read_text(encoding="utf-8") == "{roto"
    assert entorno["avisos"][-1][:2] == ("btc_rangos_estado_ilegible", "CRITICAL")


def test_no_mezcla_combinaciones_si_cambia_la_config(entorno):
    full = serie()
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(full, t))
    antes = (entorno["dir"] / "estado.json").read_text()
    entorno["cfg_path"].write_text(json.dumps(dict(entorno["cfg"], salida="S4_0.5")), encoding="utf-8")
    t2 = t + pd.Timedelta(hours=1)
    assert br.run_btc_rangos(t2, fuente(full, t2)) is None
    assert (entorno["dir"] / "estado.json").read_text() == antes
    assert entorno["avisos"][-1][0] == "btc_rangos_modo_rechazado"


def test_modo_real_se_rechaza(entorno):
    entorno["cfg_path"].write_text(json.dumps(dict(entorno["cfg"], modo="real")), encoding="utf-8")
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    assert br.run_btc_rangos(t, fuente(serie(), t)) is None
    assert not (entorno["dir"] / "estado.json").exists()


def test_estado_ida_y_vuelta_por_json():
    m = mm.Motor(cfg=mm.Config(salida="S4_0.75"), dca=lambda k: 0.0)
    m.slots[1].abierto, m.slots[1].desde, m.slots[1].qty = True, pd.Timestamp("2024-01-05 03:00"), 0.001
    m.mes["realizado"], m.ciclo, m.pendientes, m.rango = 3.5, "2024-01", [2], float("nan")
    d = json.loads(json.dumps(br.motor_a_dict(m)))
    m2 = br.motor_desde_dict(d, m.cfg, m.dca)
    assert m2.slots[1].desde == pd.Timestamp("2024-01-05 03:00") and m2.slots[1].abierto
    assert m2.mes["realizado"] == 3.5 and m2.pendientes == [2] and m2.ciclo == "2024-01"
    assert np.isnan(m2.rango)
