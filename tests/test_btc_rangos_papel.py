"""Bot BTC por rangos en papel con lotes (02/11/2026): paridad con la simulación, depósitos del
real, migración desde v0.4 y bordes de persistencia."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pkg.btc_rangos as br
from pkg import btc_rangos_lotes as ml
from pkg import btc_rangos_motor as mm

CFG = json.loads(br.CONFIG_PATH.read_text(encoding="utf-8"))
PROD = Path(__file__).parent / "datos" / "btc_rangos_v04_20261102"


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
    return {"cfg": cfg, "cfg_path": f, "avisos": avisos, "dir": tmp_path / "papel", "dep": tmp_path / "depositos.csv"}


def fuente(entorno, full, ahora):
    # devuelve también la vela en formación, como el API real
    return {"velas": lambda limit: full[full.index <= ahora.tz_localize(None)].tail(limit), "depositos": entorno["dep"]}


def depositar(entorno, t: str, lote: int, monto: float):
    """Lo que escribe el real cuando detecta un depósito en la vela `t`."""
    fila = pd.DataFrame([{"ts": pd.Timestamp(t, tz="UTC") + pd.Timedelta(minutes=1), "t": pd.Timestamp(t),
                          "lote": lote, "monto": monto, "excedente": 0.0, "precio": 0.0, "reemplaza": ""}])
    fila.to_csv(entorno["dep"], mode="a", header=not entorno["dep"].exists(), index=False)


def simular(entorno, full, desde, hasta, deps):
    """La referencia: `MotorLotes.vela` vela a vela, como `scripts/btc_rangos_lotes_sim.py`."""
    cfg = entorno["cfg"]
    datos = br.preparar(full, cfg)
    m = br.motor_nuevo(cfg)
    sl = datos[(datos.index >= desde) & (datos.index < hasta)]
    m.depositar(float(cfg["capital"]), 0, sl.index[0], float(full["close"].loc[:desde].iloc[-2]))
    for t, fila in sl.iterrows():
        m.vela(t, fila["open"], fila["high"], fila["low"], fila["close"], bool(fila["senal"]),
               float(fila["atr_prev"]), float(fila["rango_prev"]), deps.get(t, ()))
    return m


def test_paridad_con_la_simulacion_corriendo_hora_a_hora_con_huecos_y_depositos(entorno):
    full = serie()
    inicio = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    assert br.run_btc_rangos(inicio, fuente(entorno, full, inicio))["nuevo"]
    t = inicio
    fin = pd.Timestamp("2024-03-15 00:30", tz="UTC")
    huecos = {pd.Timestamp("2024-02-20 05:30", tz="UTC"), pd.Timestamp("2024-03-03 01:30", tz="UTC")}
    # el 22/02 hay un atrapado (desde el 19/02): 80 USDT lo reemplazan; 30 y 100 van en tercios
    deps = {pd.Timestamp("2024-02-22 07:00"): [(80.0, 1)], pd.Timestamp("2024-02-26 10:00"): [(30.0, 2)],
            pd.Timestamp("2024-03-08 09:00"): [(100.0, 3)]}
    while t < fin:
        t += pd.Timedelta(hours=1)
        if t in huecos:            # el server no corrió unas horas
            t += pd.Timedelta(hours=6)
        for td, lst in deps.items():
            if td == t.tz_localize(None).floor("h"):  # el real lo detecta a las :01 de esa vela
                for monto, lote in lst:
                    depositar(entorno, str(td), lote, monto)
        br.run_btc_rangos(t, fuente(entorno, full, t))
    hasta = pd.Timestamp("2024-03-15 00:00")  # última vela cerrada en la última corrida: 23:00
    m = simular(entorno, full, pd.Timestamp("2024-02-10 00:00"), hasta, deps)
    sim = pd.DataFrame(m.operaciones)
    papel = pd.read_csv(entorno["dir"] / "operaciones.csv", parse_dates=["entrada_t", "salida_t"])
    assert len(sim) > 5, "la serie sintética debe generar operaciones"
    assert m.st["reemplazos_deposito"] == 1 and m.st["n_atrapadas"] >= 2 and m.st["usdt_sale_ciclos"] > 0, m.st
    assert len(papel) == len(sim)
    assert (papel["slot"].to_numpy() == sim["slot"].to_numpy()).all()
    assert (papel["entrada_t"].to_numpy() == sim["entrada_t"].to_numpy()).all()
    assert (papel["salida_t"].to_numpy() == sim["salida_t"].to_numpy()).all()
    assert np.allclose(papel["ganancia"], sim["ganancia"])
    estado = json.loads((entorno["dir"] / "estado.json").read_text())
    m2 = ml.MotorLotes.desde_dict(estado["motor"], br.config_motor(entorno["cfg"]))
    assert m2.holding_btc == pytest.approx(m.holding_btc)
    assert m2.st["depositado"] == pytest.approx(m.st["depositado"]) == pytest.approx(186.65 + 210)
    assert m2.st == pytest.approx({k: v for k, v in m.st.items()}, rel=1e-9)
    assert m2.bench_btc > 0 and sorted(estado["lotes_aplicados"]) == [1, 2, 3]
    assert m2.n_cierres == 1      # cruzó el 03/03: sólo el corte contable
    mensual = pd.read_csv(entorno["dir"] / "mensual.csv")
    assert len(mensual) == 1 and mensual["depositado"].iloc[0] == pytest.approx(186.65 + 110)
    cats = [a[0] for a in entorno["avisos"]]
    assert "btc_rangos_operaciones" in cats and "btc_rangos_cierre_mensual" in cats
    assert "btc_rangos_resumen" in cats


def test_deposito_detectado_tarde_entra_en_la_vela_siguiente_y_una_sola_vez(entorno):
    full = serie()
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(entorno, full, t))
    for _ in range(5):
        t += pd.Timedelta(hours=1)
        br.run_btc_rangos(t, fuente(entorno, full, t))
    depositar(entorno, "2024-02-10 02:00", 1, 90.0)  # el papel ya pasó esa vela
    for _ in range(3):
        t += pd.Timedelta(hours=1)
        br.run_btc_rangos(t, fuente(entorno, full, t))
    estado = json.loads((entorno["dir"] / "estado.json").read_text())
    m = ml.MotorLotes.desde_dict(estado["motor"], br.config_motor(entorno["cfg"]))
    assert estado["lotes_aplicados"] == [1]
    assert m.st["depositado"] == pytest.approx(186.65 + 90.0)


def test_estado_ilegible_no_escribe_y_avisa(entorno):
    full = serie()
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(entorno, full, t))
    f = entorno["dir"] / "estado.json"
    f.write_text("{roto", encoding="utf-8")
    t2 = t + pd.Timedelta(hours=1)
    assert br.run_btc_rangos(t2, fuente(entorno, full, t2)) is None
    assert f.read_text(encoding="utf-8") == "{roto"
    assert entorno["avisos"][-1][:2] == ("btc_rangos_estado_ilegible", "CRITICAL")


def test_no_mezcla_combinaciones_si_cambia_la_config(entorno):
    full = serie()
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(entorno, full, t))
    antes = (entorno["dir"] / "estado.json").read_text()
    entorno["cfg_path"].write_text(json.dumps(dict(entorno["cfg"], ciclos_por_lote=4)), encoding="utf-8")
    t2 = t + pd.Timedelta(hours=1)
    assert br.run_btc_rangos(t2, fuente(entorno, full, t2)) is None
    assert (entorno["dir"] / "estado.json").read_text() == antes
    assert entorno["avisos"][-1][0] == "btc_rangos_modo_rechazado"


def test_modo_real_se_rechaza(entorno):
    entorno["cfg_path"].write_text(json.dumps(dict(entorno["cfg"], modo="real")), encoding="utf-8")
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    assert br.run_btc_rangos(t, fuente(entorno, serie(), t)) is None
    assert not (entorno["dir"] / "estado.json").exists()


def test_estado_ida_y_vuelta_por_json():
    m = br.motor_nuevo(CFG)
    m.depositar(300.0, 0, pd.Timestamp("2024-01-05 00:00"), 40_000.0)
    m.slots[1].abierto, m.slots[1].desde, m.slots[1].qty, m.slots[1].costo = True, pd.Timestamp("2024-01-05 03:00"), 0.001, 99.0
    m.atrapado[1], m.ciclo, m.pendientes, m.rango, m.reserva = True, "2024-01", [2], float("nan"), 0.4
    m.corte_mensual(pd.Timestamp("2024-01-03"), 39_000.0)
    d = json.loads(json.dumps(m.a_dict()))
    m2 = ml.MotorLotes.desde_dict(d, m.cfg)
    assert m2.slots[1].desde == pd.Timestamp("2024-01-05 03:00") and m2.slots[1].abierto
    assert m2.tr[0][0]["t0"] == pd.Timestamp("2024-01-05 00:00") and m2.tr[0][0]["usdt"] == pytest.approx(100.0)
    assert m2.atrapado == [False, True, False, False] and m2.pendientes == [2] and m2.ciclo == "2024-01"
    assert m2.max_ciclos == 3 and m2.caida == 0.10 and m2.dias == 30
    assert m2.corte_previo == m.corte_previo and m2.st == m.st and m2.reserva == 0.4
    assert m2.usdt_bot() == pytest.approx(m.usdt_bot()) and np.isnan(m2.rango)


def test_mensaje_de_inicio_sin_guiones_bajos(entorno):
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    br.run_btc_rangos(t, fuente(entorno, serie(), t))
    cat, _, kw = entorno["avisos"][-1]
    assert cat == "btc_rangos_inicio"
    assert "_" not in kw["mensaje_md"].replace("_Modo papel: sin dinero real._", "")
    assert "compra tras caer 0,5 × rango diario" in kw["mensaje_md"]
    assert "vende en +0,75 × rango" in kw["mensaje_md"]
    assert "lotes de 3 ciclos" in kw["mensaje_md"]


def test_migra_el_estado_v04_de_prod(entorno):
    """La copia del estado del papel en prod del 02/11 (v0.4, día 28): se migra sin perder los
    slots abiertos, con respaldo, y sigue corriendo con las reglas nuevas."""
    shutil.copytree(PROD / "papel", entorno["dir"])
    antes = json.loads((entorno["dir"] / "estado.json").read_text())
    full = serie()
    full.index = pd.date_range("2026-09-25 00:00", periods=len(full), freq="h")  # termina el 08/12
    t = pd.Timestamp("2026-11-02 18:30", tz="UTC")
    r = br.run_btc_rangos(t, fuente(entorno, full, t))
    assert r is not None and not r["nuevo"]
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["reglas"] == "lotes" and est["combo"]["dia_cierre"] == 3 and Path(est["respaldo_v04"]).exists()
    assert json.loads((Path(est["respaldo_v04"]) / "estado.json").read_text()) == antes
    m = ml.MotorLotes.desde_dict(est["motor"], br.config_motor(entorno["cfg"]))
    viejos = antes["motor"]["slots"]
    for i in range(3):
        assert m.slots[i].abierto == viejos[i]["abierto"] and m.slots[i].qty == viejos[i]["qty"]
        assert m.tr[i][0]["lote"] == 0 and m.tr[i][0]["ciclos"] == 0
    assert m.reserva == pytest.approx(antes["motor"]["usdt_para_holding"])
    assert m.st["realizado"] == pytest.approx(pd.read_csv(PROD / "papel" / "operaciones.csv")["ganancia"].sum())
    assert m.bench_btc == mm._redondear(186.65 * 0.999 / (est["precio_inicio"] * 1.0005))
    assert m.ciclo == "2026-10"
    for f in entorno["dir"].glob("*.csv"):  # cada CSV se puede leer entero
        pd.read_csv(f)
    assert "latente" in pd.read_csv(entorno["dir"] / "equity.csv")
