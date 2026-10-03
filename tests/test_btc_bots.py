"""Cuatro bots sobre BTC con libros virtuales (02/10/2026), modo papel."""
import json

import pandas as pd
import pytest

import pkg.btc_bots as bb

T0 = pd.Timestamp("2026-10-03 00:02", tz="UTC")
P0 = 50_000.0


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    cfg = json.loads(bb.CONFIG_PATH.read_text(encoding="utf-8"))
    cfg.update(capital_total=400.0, modo="papel", enabled=True)
    p = tmp_path / "btc_bots.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(bb, "CONFIG_PATH", p)
    monkeypatch.setattr(bb, "OUT_DIR", tmp_path / "btc_bots")
    eventos = []
    monkeypatch.setattr(bb, "_emitir", lambda cat, sev, **kw: eventos.append((cat, sev, kw)))
    objetivos = {"T1": -1.0, "T3": 1.0, "R1": 0.0}
    monkeypatch.setattr(bb, "posiciones", lambda v, regla, solo_long: pd.Series(objetivos[regla], index=v.index))
    return tmp_path, eventos, objetivos, cfg


def velas(interval, limit, ahora):
    """Velas al estilo BingX: la última está EN FORMACIÓN."""
    paso = pd.Timedelta(hours=bb.HORAS[interval])
    ult = ahora.floor("D") if interval == "1d" else ahora.floor("4h")
    fechas = pd.date_range(end=ult, periods=min(limit, 200), freq=paso)
    return [{"date": d, "open": P0, "high": P0, "low": P0, "close": P0} for d in fechas]


def fuentes(ahora, precio=P0, funding=None):
    return {"velas": lambda interval, limit: velas(interval, limit, ahora),
            "precio": lambda: precio, "funding": lambda: list(funding or [])}


def estado(tmp):
    return json.loads((tmp / "btc_bots/estado.json").read_text())


def test_primer_ciclo_abre_segun_las_reglas(entorno):
    tmp, eventos, _, _ = entorno
    r = bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    libros = estado(tmp)["bots"]
    assert libros["A"]["lado"] == -1 and libros["B"]["lado"] == 1 and libros["D"]["lado"] == 0
    assert libros["C"]["lado"] == -1 and libros["C"]["spot_qty"] == libros["C"]["qty"] == 0.001   # 0,5 × 100 / 50.000
    assert libros["A"]["qty"] == 0.002
    assert r["posicion_neta_btc"] == pytest.approx(-0.002 + 0.002 - 0.001)
    # decide con la última vela CERRADA, no con la que está en formación
    assert pd.Timestamp(libros["A"]["ultima_barra"]) == T0.floor("D") - pd.Timedelta(days=1)
    assert pd.Timestamp(libros["B"]["ultima_barra"]) == T0.floor("4h") - pd.Timedelta(hours=4)
    assert [e[0] for e in eventos] == ["btc_bots_inicio", "btc_bots_operaciones"]
    ops = pd.read_csv(tmp / "btc_bots/operaciones.csv")
    assert list(ops.bot) == ["A", "B", "C"]


def test_no_decide_dos_veces_la_misma_vela_y_sobrevive_al_reinicio(entorno):
    tmp, _, objetivos, _ = entorno
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    objetivos.update(T1=1.0, T3=-1.0)          # la regla cambiaría, pero la vela es la misma
    t1 = T0 + pd.Timedelta(hours=1)
    r = bb.run_btc_bots(ahora=t1, fuentes=fuentes(t1))
    assert r["operaciones"] == []
    t4 = T0 + pd.Timedelta(hours=4)             # cierra una vela de 4 h: sólo decide B
    r = bb.run_btc_bots(ahora=t4, fuentes=fuentes(t4))
    assert [o["bot"] for o in r["operaciones"]] == ["B"]
    assert estado(tmp)["bots"]["A"]["lado"] == -1 and estado(tmp)["bots"]["B"]["lado"] == -1


def test_dar_vuelta_contabiliza_precio_y_comisiones(entorno):
    tmp, _, objetivos, cfg = entorno
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    objetivos["T1"] = 1.0
    t1, p1 = T0 + pd.Timedelta(days=1), 45_000.0
    r = bb.run_btc_bots(ahora=t1, fuentes=fuentes(t1, precio=p1))
    op_a = [o for o in r["operaciones"] if o["bot"] == "A"][0]
    c = cfg["costo_lado"]
    esperado = 0.002 * (P0 - p1) - 0.002 * P0 * c - 0.002 * p1 * c          # short ganó 10
    assert op_a["resultado_cerrada"] == pytest.approx(esperado)
    a = estado(tmp)["bots"]["A"]
    assert a["lado"] == 1 and a["precio_entrada"] == p1
    assert a["qty"] == bb.redondear_qty((100 + esperado) / p1)               # compone su capital
    assert bb.capital(a, p1) == pytest.approx(100 + esperado - a["qty"] * p1 * c)


def test_funding_long_paga_y_short_cobra_una_sola_vez(entorno):
    tmp, _, _, _ = entorno
    ms0 = int(T0.timestamp() * 1000)
    previo = {"ms": ms0 - 3_600_000, "tasa": 0.01, "precio": P0}   # anterior al arranque: no se cobra
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0, funding=[previo]))
    antes = estado(tmp)["bots"]
    ev = {"ms": ms0 + 1_000, "tasa": 0.0001, "precio": 52_000.0}
    t1 = T0 + pd.Timedelta(hours=1)
    bb.run_btc_bots(ahora=t1, fuentes=fuentes(t1, funding=[previo, ev]))
    bb.run_btc_bots(ahora=t1 + pd.Timedelta(hours=1), fuentes=fuentes(t1, funding=[previo, ev]))  # no se repite
    despues = estado(tmp)["bots"]
    pago = 0.002 * 52_000 * 0.0001
    assert despues["B"]["efectivo"] == pytest.approx(antes["B"]["efectivo"] - pago)     # long paga
    assert despues["A"]["efectivo"] == pytest.approx(antes["A"]["efectivo"] + pago)     # short cobra
    assert despues["C"]["funding_neto"] == pytest.approx(pago / 2)
    assert despues["D"]["efectivo"] == antes["D"]["efectivo"]


def test_carry_no_depende_del_precio(entorno):
    tmp, _, _, cfg = entorno
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    c = estado(tmp)["bots"]["C"]
    costos = 0.001 * P0 * (cfg["costo_lado"] + cfg["costo_spot_lado"])
    for p in (20_000.0, 50_000.0, 90_000.0):
        assert bb.capital(c, p) == pytest.approx(100 - costos)


def test_corte_cierra_y_apaga_el_bot(entorno):
    tmp, eventos, objetivos, _ = entorno
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    t1 = T0 + pd.Timedelta(hours=1)
    r = bb.run_btc_bots(ahora=t1, fuentes=fuentes(t1, precio=P0 * 0.6))      # B long pierde 40%
    corte = [o for o in r["operaciones"] if o["bot"] == "B"]
    assert len(corte) == 1 and corte[0]["accion"].startswith("CORTE")
    b = estado(tmp)["bots"]["B"]
    assert b["apagado"] and b["lado"] == 0
    objetivos["T3"] = -1.0
    t4 = T0 + pd.Timedelta(hours=4)
    r = bb.run_btc_bots(ahora=t4, fuentes=fuentes(t4, precio=P0 * 0.6))
    assert all(o["bot"] != "B" for o in r["operaciones"])


def test_resumen_diario_una_vez_por_dia(entorno):
    _, eventos, _, _ = entorno
    bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0))
    for h in (1, 5, 23, 24, 25):
        t = T0 + pd.Timedelta(hours=h)
        bb.run_btc_bots(ahora=t, fuentes=fuentes(t))
    assert [e[0] for e in eventos].count("btc_bots_resumen") == 1


def test_modo_real_se_rechaza_sin_tocar_nada(entorno):
    tmp, eventos, _, cfg = entorno
    cfg["modo"] = "real"
    bb.CONFIG_PATH.write_text(json.dumps(cfg))
    bb._MODO_AVISADO.clear()
    assert bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0)) is None
    assert bb.run_btc_bots(ahora=T0, fuentes=fuentes(T0)) is None
    assert not (tmp / "btc_bots").exists()
    assert [e[0] for e in eventos] == ["btc_bots_modo_rechazado"]


def test_nunca_levanta_excepcion(entorno):
    tmp, _, _, _ = entorno
    def explota():
        raise RuntimeError("API caída")
    f = fuentes(T0)
    f["precio"] = explota
    assert bb.run_btc_bots(ahora=T0, fuentes=f) is None
    f = fuentes(T0, precio=None)
    assert bb.run_btc_bots(ahora=T0, fuentes=f) is None
    assert not (tmp / "btc_bots/estado.json").exists()


def test_velas_cerradas_descarta_la_que_esta_en_formacion():
    v = bb.velas_cerradas(velas("4h", 50, T0), 4, T0)
    assert v.index[-1] == T0.floor("4h") - pd.Timedelta(hours=4)
    assert len(v) == 49


def test_redondeo_al_paso_de_btc():
    assert bb.redondear_qty(0.00054) == 0.0005
    assert bb.redondear_qty(0.00056) == 0.0006
    assert bb.redondear_qty(0.00004) == 0.0
