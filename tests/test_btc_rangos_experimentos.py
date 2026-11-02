"""Experimentos en papel junto al real (02/11/2026): lotes de 1 ciclo y la regla v0.4 con los
depósitos del real. Paridad con la simulación, silencio en Telegram y la tabla del tablero."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pkg.btc_rangos as br
import pkg.btc_rangos_experimentos as bx
import pkg.btc_rangos_v04 as v04
from pkg import btc_rangos_lotes as ml
from pkg import btc_rangos_motor as mm
from pkg import btc_rangos_reglas as rr

PROD = Path(__file__).parent / "datos" / "btc_rangos_v04_20261102"
CFG_B1 = json.loads(bx.EXPERIMENTOS["exp_b1"]["config"].read_text(encoding="utf-8"))
CFG_V04 = json.loads(bx.EXPERIMENTOS["exp_v04"]["config"].read_text(encoding="utf-8"))


def serie(dias=75, semilla=7, desde="2024-01-01"):
    idx = pd.date_range(desde, periods=dias * 24, freq="h")
    rng = np.random.default_rng(semilla)
    c = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, len(idx)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, len(idx)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    avisos = []
    for mod in (br, v04):
        monkeypatch.setattr(mod, "_AVISADO", set())
        monkeypatch.setattr(mod, "_emitir", lambda cat, sev, **kw: avisos.append((cat, sev, kw)))
    # las configs del repo, pero en tmp: el job escribe el estado en base/<clave>
    exps = {}
    for clave, e in bx.EXPERIMENTOS.items():
        f = tmp_path / f"{clave}.json"
        shutil.copy(e["config"], f)
        exps[clave] = dict(e, config=f)
    monkeypatch.setattr(bx, "EXPERIMENTOS", exps)
    monkeypatch.setattr(v04, "CONFIG_PATH", v04.CONFIG_PATH)  # el job los pisa: que el test los restaure
    monkeypatch.setattr(v04, "OUT_DIR", v04.OUT_DIR)
    base = tmp_path / "btc_rangos"
    (base / "real").mkdir(parents=True)
    return {"base": base, "avisos": avisos, "dep": base / "real" / "depositos.csv", "exps": exps}


def fuente(entorno, full, ahora):
    return {"velas": lambda limit: full[full.index <= ahora.tz_localize(None)].tail(limit), "depositos": entorno["dep"]}


def depositar(entorno, t: str, lote: int, monto: float, precio: float = 0.0):
    fila = pd.DataFrame([{"ts": pd.Timestamp(t, tz="UTC") + pd.Timedelta(minutes=1), "t": pd.Timestamp(t),
                          "lote": lote, "monto": monto, "excedente": 0.0, "precio": precio, "reemplaza": ""}])
    fila.to_csv(entorno["dep"], mode="a", header=not entorno["dep"].exists(), index=False)


def correr(entorno, full, desde, hasta):
    t = desde
    while t <= hasta:
        bx.run_btc_rangos_experimentos(t, fuente(entorno, full, t), base=entorno["base"])
        t += pd.Timedelta(hours=1)


def test_v04_paridad_con_la_simulacion_y_el_deposito_entra_en_el_cierre(entorno):
    """El depósito del 15/02 queda en USDT (cuenta en el patrimonio) y entra recién en el cierre
    del 03/03, como en la v0.4 real. Todo igual a `mm.simular` con ese DCA en el primer cierre."""
    full = serie()
    depositar(entorno, "2024-02-15 10:00", 1, 120.0)
    depositar(entorno, "2024-03-05 10:00", 2, 50.0)   # después del cierre: espera al siguiente
    correr(entorno, full, pd.Timestamp("2024-02-10 00:30", tz="UTC"), pd.Timestamp("2024-02-20 00:30", tz="UTC"))
    d = entorno["base"] / "exp_v04"
    eq = pd.read_csv(d / "equity.csv")
    assert eq["usdt_espera"].iloc[-1] == 120.0
    u = eq.iloc[-1]
    sin_espera = u["patrimonio_btc"] - 120.0 / u["precio"]
    assert sin_espera == pytest.approx(u["holding_btc"] + u["btc_slots"] + u["usdt_bot"] / u["precio"], abs=1e-4)
    correr(entorno, full, pd.Timestamp("2024-02-20 01:30", tz="UTC"), pd.Timestamp("2024-03-15 00:30", tz="UTC"))

    cfg = dict(CFG_V04)
    datos = v04.preparar(full, cfg)
    r = mm.simular(full, datos["senal"], rr.atr(full), v04.config_motor(cfg), lambda k: 120.0 if k == 1 else 0.0,
                   pd.Timestamp("2024-02-10 00:00"), pd.Timestamp("2024-03-15 00:00"), rango=datos["rango"])
    sim = pd.DataFrame(r.motor.operaciones)
    papel = pd.read_csv(d / "operaciones.csv", parse_dates=["entrada_t", "salida_t"])
    assert len(sim) > 5 and len(papel) == len(sim)
    assert (papel["entrada_t"].to_numpy() == sim["entrada_t"].to_numpy()).all()
    assert np.allclose(papel["ganancia"], sim["ganancia"])
    mensual = pd.read_csv(d / "mensual.csv")
    assert len(mensual) == 1 and mensual["dca"].iloc[0] == 120.0
    est = json.loads((d / "estado.json").read_text())
    assert est["lotes_aplicados"] == [1]
    assert est["motor"]["holding_btc"] == pytest.approx(r.motor.holding_btc)
    assert pd.read_csv(d / "equity.csv")["usdt_espera"].iloc[-1] == 50.0
    # los experimentos no mandan Telegram
    assert not [a for a in entorno["avisos"] if a[1] == "INFO"]


def test_b1_corre_en_su_carpeta_y_cada_tramo_hace_un_solo_ciclo(entorno):
    full = serie()
    correr(entorno, full, pd.Timestamp("2024-02-10 00:30", tz="UTC"), pd.Timestamp("2024-03-15 00:30", tz="UTC"))
    d = entorno["base"] / "exp_b1"
    est = json.loads((d / "estado.json").read_text())
    assert est["combo"]["ciclos_por_lote"] == 1
    ops = pd.read_csv(d / "operaciones.csv")
    m = ml.MotorLotes.desde_dict(est["motor"], br.config_motor(CFG_B1))
    # con 1 ciclo, cada slot vende una vez y su capital pasa a holding: 3 ventas y no hay más
    assert len(ops) == 3 and m.holding_btc > 0
    assert not (entorno["base"] / "papel").exists()          # no toca la sombra
    assert not [a for a in entorno["avisos"] if a[1] == "INFO"]


def test_uno_que_falla_no_frena_al_otro(entorno):
    full = serie()
    (entorno["base"] / "exp_v04").mkdir()
    (entorno["base"] / "exp_v04" / "estado.json").write_text("{roto", encoding="utf-8")
    t = pd.Timestamp("2024-02-10 00:30", tz="UTC")
    r = bx.run_btc_rangos_experimentos(t, fuente(entorno, full, t), base=entorno["base"])
    assert r["exp_v04"] is None and r["exp_b1"]["nuevo"]
    assert ("btc_rangos_estado_ilegible", "CRITICAL") in [(a[0], a[1]) for a in entorno["avisos"]]


def test_arrancan_del_estado_de_prod_del_02_11(entorno):
    """Como se siembran en el deploy: exp_b1 con la copia del papel v0.4 (migra igual que la
    sombra) y exp_v04 con el respaldo de las 17:00, que cierra el 03/11 00:00 al ponerse al día."""
    base = entorno["base"]
    shutil.copytree(PROD / "papel", base / "exp_b1")
    shutil.copytree(PROD / "papel", base / "exp_v04")
    full = serie(desde="2026-09-25")  # termina el 08/12
    t = pd.Timestamp("2026-11-03 02:30", tz="UTC")
    r = bx.run_btc_rangos_experimentos(t, fuente(entorno, full, t), base=base)
    assert r["exp_b1"] is not None and r["exp_v04"] is not None
    b1 = json.loads((base / "exp_b1" / "estado.json").read_text())
    assert b1["reglas"] == "lotes" and b1["combo"]["ciclos_por_lote"] == 1 and Path(b1["respaldo_v04"]).exists()
    assert Path(b1["respaldo_v04"]).name.startswith("exp_b1_v04_")
    e4 = json.loads((base / "exp_v04" / "estado.json").read_text())
    assert e4["ultima_barra"] == "2026-11-03 01:00:00"
    mensual = pd.read_csv(base / "exp_v04" / "mensual.csv")
    assert list(mensual["ciclo"]) == ["2026-10"] and pd.Timestamp(mensual["cierre"].iloc[0]) == pd.Timestamp("2026-11-03")
    assert e4["motor"]["holding_btc"] > 0  # los abiertos quedaron atrapados y pasaron a holding


def test_comparacion_valua_todo_al_mismo_precio_contra_un_dca_propio(entorno):
    base = entorno["base"]
    for clave, btc_slots, holding, usdt, precio in (("real", 0.001, 0.0005, 60.0, 80_000.0),
                                                     ("papel", 0.001, 0.0005, 61.0, 80_000.0),
                                                     ("exp_b1", 0.0, 0.0020, 20.0, 79_000.0)):
        d = base / clave
        d.mkdir(exist_ok=True)
        pat = btc_slots + holding + usdt / precio
        pd.DataFrame([{"ts": "2026-11-10 10:01", "precio": precio, "btc_slots": btc_slots, "holding_btc": holding,
                       "usdt_bot": usdt, "patrimonio_btc": pat, "atrapados": 0}]).to_csv(d / "equity.csv", index=False)
        pd.DataFrame([{"slot": 1, "entrada_t": "2026-11-04 01:00", "salida_t": "2026-11-05 01:00", "ganancia": 0.5},
                      {"slot": 1, "entrada_t": "2026-10-26 01:00", "salida_t": "2026-10-27 01:00", "ganancia": 0.4}]
                     ).to_csv(d / "operaciones.csv", index=False)
        (d / "estado.json").write_text(json.dumps({"inicio": "2026-10-25T00:01:00+00:00",
                                                   "inicio_lotes": "2026-11-03T15:00:00+00:00",
                                                   "precio_inicio": 80877.24, "combo": {"capital": 186.65}}))
    depositar(entorno, "2026-11-15 10:00", 1, 100.0, precio=75_000.0)
    c = bx.comparacion(base)
    assert c["precio"] == 80_000.0
    dca = 186.65 * 0.999 / (80877.24 * 1.0005) + 100.0 * 0.999 / (75_000.0 * 1.0005)
    assert c["dca_puro_btc"] == pytest.approx(dca)
    f = {x["clave"]: x for x in c["filas"]}
    assert f["exp_b1"]["patrimonio_btc"] == pytest.approx(0.002 + 20.0 / 80_000.0)  # revaluado al precio del real
    assert f["real"]["vs_real_pct"] == pytest.approx(0.0)
    assert f["real"]["ventas"] == 1 and f["real"]["realizado"] == pytest.approx(0.5)  # sólo desde el arranque
    assert not f["exp_v04"]["corre"]
