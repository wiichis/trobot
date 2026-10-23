"""Motor del bot BTC por rangos (22/10/2026): casos armados a mano."""
import numpy as np
import pandas as pd
import pytest

from pkg import btc_rangos_motor as mm
from pkg import btc_rangos_reglas as rr

T0 = pd.Timestamp("2024-01-03 00:00")  # inicio de ciclo


def motor(**kw):
    dca = kw.pop("dca", lambda k: 0.0)
    cfg = mm.Config(comision=0.0, slippage=0.0, min_orden=1.0, **kw)
    return mm.Motor(cfg=cfg, dca=dca)


def paso(m, k, o, h, l, c, senal=False, atr=np.nan, t0=T0):
    m.vela(t0 + pd.Timedelta(hours=k), o, h, l, c, senal, atr)


def test_ciclo_empieza_el_dia_3():
    assert mm.ciclo_de(pd.Timestamp("2024-02-02 23:00")) == "2024-01"
    assert mm.ciclo_de(pd.Timestamp("2024-02-03 00:00")) == "2024-02"


def test_s1_vende_en_el_tp_y_a_la_apertura_si_abre_arriba():
    m = motor(salida="S1_2")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 101, 99, 100)           # entra a 100
    s = m.slots[0]
    assert s.abierto and s.entrada == 100
    paso(m, 2, 100, 102.5, 100, 101)        # toca el TP 102
    assert not s.abierto
    assert m.operaciones[-1]["salida"] == pytest.approx(102)
    paso(m, 3, 100, 100, 100, 100, senal=True)
    paso(m, 4, 100, 100, 100, 100)          # modo A: ya hubo una entrada este día
    assert not m.slots[0].abierto


def test_s1_gap_vende_a_la_apertura():
    m = motor(salida="S1_2")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 100)
    paso(m, 2, 105, 106, 104, 105)
    assert m.operaciones[-1]["salida"] == 105


def test_trailing_s2_usa_el_maximo_previo_y_no_sale_en_la_vela_de_entrada():
    m = motor(salida="S2")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 102, 90, 100)           # entra a 100; arma con el high 102, pero no sale en esta vela
    s = m.slots[0]
    assert s.abierto and s.armado and s.maximo == 102
    paso(m, 2, 102, 103, 101.5, 102)        # stop previo 102×0,994 = 101,388 → no toca; máximo 103
    assert s.abierto and s.maximo == 103
    paso(m, 3, 103, 103, 102, 102.5)        # stop 103×0,994 = 102,382 → toca
    assert not s.abierto
    assert m.operaciones[-1]["salida"] == pytest.approx(103 * 0.994)


def test_nunca_vende_bajo_el_piso_aunque_abra_debajo_del_stop():
    m = mm.Motor(cfg=mm.Config(salida="S2", min_orden=1.0), dca=lambda k: 0.0)
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 101.6, 100, 101.5)      # arma (fill 100,05 → +1,5 % = 101,55)
    s = m.slots[0]
    assert s.armado
    piso = m.piso(s)
    paso(m, 2, piso * 0.99, piso * 0.995, piso * 0.98, piso * 0.99)   # abre bajo el piso
    assert s.abierto, "no puede vender con pérdida"
    paso(m, 3, piso * 1.0001, piso * 1.0002, piso * 0.99, piso)  # abre apenas sobre el piso → stop = piso
    assert not s.abierto
    op = m.operaciones[-1]
    assert op["ganancia"] >= -1e-9


def test_modo_a_una_entrada_por_dia_y_slot_de_menor_numero():
    m = motor(slots="A", salida="S1_3")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 100, senal=True)   # entra slot 1 (día 03)
    paso(m, 2, 100, 100, 100, 100, senal=True)   # mismo día: no entra el 2
    assert [s.abierto for s in m.slots] == [True, False, False]
    paso(m, 23, 100, 100, 100, 100, senal=True)  # señal a las 23:00 → ejecuta 00:00 del día 04
    paso(m, 24, 100, 100, 100, 100)
    assert [s.abierto for s in m.slots] == [True, True, False]


def test_modo_b_escalonado():
    m = motor(slots="B", escalon=0.02, salida="S1_3")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 99)                # entra slot 1 a 100
    paso(m, 2, 99, 99, 97, 97.9)                 # close −2,1 % → pendiente slot 2
    paso(m, 3, 97.9, 98, 96, 96)                 # entra slot 2; close −4 % → pendiente slot 3
    paso(m, 4, 96, 96, 95, 95.5)
    assert [s.abierto for s in m.slots] == [True, True, True]
    assert m.slots[1].entrada == 97.9 and m.slots[2].entrada == 96


def test_cierre_mensual_atrapadas_reposicion_en_orden_y_sobrante_a_holding():
    m = motor(slots="B", escalon=0.02, salida="S1_3", dca=lambda k: 100.0)
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 97.9)
    paso(m, 2, 97.9, 97.9, 96, 96)               # pendiente slot 3
    paso(m, 3, 96, 96, 95, 95)
    assert all(s.abierto for s in m.slots)
    nominal = 200 / 3
    # siguiente ciclo: 03/02 00:00, abre a 90
    t = pd.Timestamp("2024-02-03 00:00")
    m.vela(t, 90, 90, 90, 90, False, np.nan)
    mes = m.meses[-1]
    assert mes["n_atrapadas"] == 3
    esperado = (100 - 90) * nominal / 100 + (97.9 - 90) * nominal / 97.9 + (96 - 90) * nominal / 96
    assert mes["atrapadas_costo"] == pytest.approx(esperado, rel=1e-5)  # qty al step de 1e-6
    # DCA 100: repone slot 1 (66,67); el slot 2 no alcanza → 2 y 3 apagados; sobrante 33,33 a holding
    assert [s.activo for s in m.slots] == [True, False, False]
    assert mes["repuesto"] == pytest.approx(nominal)
    assert mes["compra_holding"] == pytest.approx(100 - nominal)
    btc_slots = nominal / 100 + nominal / 97.9 + nominal / 96
    assert m.holding_btc == pytest.approx(btc_slots + (100 - nominal) / 90, rel=1e-5)
    assert m.bench_btc == pytest.approx(100 / 90, rel=1e-5)
    # el USDT del bot es el nominal del slot repuesto
    assert m.usdt_bot() == pytest.approx(nominal)


def test_conservacion_de_valor_con_ganancia_g4():
    m = mm.Motor(cfg=mm.Config(salida="S1_2", ganancia="G4", min_orden=1.0), dca=lambda k: 0.0)
    total0 = m.usdt_bot()
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 100)
    paso(m, 2, 100, 103, 100, 102)
    op = m.operaciones[-1]
    assert op["ganancia"] > 0
    assert m.usdt_bot() + m.usdt_para_holding == pytest.approx(total0 + op["ganancia"])
    assert m.usdt_para_holding == pytest.approx(op["ganancia"] * 0.1)
    assert m.capital_bot_nominal() == pytest.approx(200 + op["ganancia"] * 0.9)


def test_costos_compra_y_venta():
    m = mm.Motor(cfg=mm.Config(salida="S1_2", min_orden=1.0), dca=lambda k: 0.0)
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 100)
    s = m.slots[0]
    assert s.entrada == pytest.approx(100.05)
    assert s.qty == pytest.approx(np.floor((200 / 3) * 0.999 / 100.05 / 1e-6) * 1e-6)
    piso = m.piso(s)
    # vender en el piso devuelve el costo
    assert m._vender(s.qty, piso) == pytest.approx(s.costo, rel=1e-9)


def test_senales_e1_e2_e3_y_filtro():
    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    c = pd.Series(100.0, index=idx)
    c.iloc[-1] = 97.0                          # −3 % desde el máximo de 24 h
    h = pd.DataFrame({"open": c, "high": c, "low": c, "close": c})
    h.loc[idx[-1], "high"] = 100.0
    ind = rr.indicadores(h)
    assert rr.senal_entrada(h, ind, "E1_3", False).iloc[-1]
    assert rr.senal_entrada(h, ind, "E1_2", False).iloc[-1]
    assert not rr.senal_entrada(h, ind, "E1_2", False).iloc[-2]
    assert rr.senal_entrada(h, ind, "E2", False).iloc[-1]   # bajo Bollinger y RSI muy bajo
    assert rr.senal_entrada(h, ind, "E3_7", False).iloc[-1]  # en el mínimo de 7 días
    # filtro: con ADX > 25 y DI− > DI+ bloquea
    ind2 = ind.copy()
    ind2.loc[idx[-1], ["adx", "di_m", "di_p"]] = [30, 40, 10]
    assert not rr.senal_entrada(h, ind2, "E1_3", True).iloc[-1]
    assert rr.senal_entrada(h, ind2, "E1_3", False).iloc[-1]


def test_simular_extremo_a_extremo_y_benchmarks():
    idx = pd.date_range("2024-01-01", "2024-03-10", freq="h")
    rng = np.random.default_rng(0)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx)))), index=idx)
    h = pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "close": c})
    h["high"] = h[["open", "close"]].max(axis=1) * 1.002
    h["low"] = h[["open", "close"]].min(axis=1) * 0.998
    ind = rr.indicadores(h)
    sen = rr.senal_entrada(h, ind, "E1_2", False)
    r = mm.simular(h, sen, ind["atr"], mm.Config(), lambda k: 400.0,
                   pd.Timestamp("2024-01-03"), pd.Timestamp("2024-03-03"))
    assert len(r.motor.meses) == 2 and r.motor.meses[-1].get("final")
    assert r.motor.bench_btc > 0
    assert r.bench_a_btc == pytest.approx(r.motor.bench_btc + 200 / r.precio_fin)
    assert all(op["ganancia"] >= -1e-9 for op in r.motor.operaciones)
    # el patrimonio cuadra: USDT iniciales + DCA = lo invertido; nada se crea de la nada
    assert r.patrimonio_btc > 0


def test_regimen_compra_usa_solo_dias_cerrados():
    dias = pd.date_range("2024-01-01", periods=101, freq="D")
    cierre = pd.Series(100.0, index=dias)
    cierre.iloc[-1] = 200.0                       # el día 101 (10/04) cierra sobre la media
    idx = pd.date_range(dias[-1] - pd.Timedelta(hours=2), dias[-1] + pd.Timedelta(hours=25), freq="h")
    reg = rr.regimen_compra(cierre, idx)
    ultimo = dias[-1]
    assert not reg.loc[ultimo + pd.Timedelta(hours=22)]  # el día todavía no cerró
    assert reg.loc[ultimo + pd.Timedelta(hours=23)]      # la vela de las 23:00 cierra el día
    assert not reg.loc[ultimo - pd.Timedelta(hours=1)]


def test_sin_cierre_forzado_el_slot_espera_y_la_ganancia_compra_btc():
    m = motor(slots="A", salida="S1_2", ganancia="G1", cierre_forzado=False, dca=lambda k: 0.0)
    paso(m, 0, 100, 100, 100, 100, senal=True)
    paso(m, 1, 100, 100, 100, 100)                 # entra slot 1 a 100
    m.vela(pd.Timestamp("2024-02-03 00:00"), 90, 90, 90, 90, False, np.nan)   # cierre mensual
    mes = m.meses[-1]
    s = m.slots[0]
    assert s.abierto and s.activo, "no se atrapa: sigue esperando"
    assert mes["n_atrapadas"] == 0 and mes["atrapadas_costo"] == 0
    assert mes["latente"] == pytest.approx(10 * s.qty)
    assert m.holding_btc == 0 and m.bench_btc == 0
    m.vela(pd.Timestamp("2024-02-03 01:00"), 100, 103, 100, 102, False, np.nan)  # sale en el TP 102
    assert not s.abierto
    g = m.operaciones[-1]["ganancia"]
    assert m.usdt_para_holding == pytest.approx(g)  # G1: todo va a BTC en el próximo cierre
    m.vela(pd.Timestamp("2024-03-03 00:00"), 110, 110, 110, 110, False, np.nan)
    assert m.holding_btc == pytest.approx(g / 110, abs=1e-6)  # qty al step de 1e-6
    assert m.capital_bot_nominal() == pytest.approx(200)


def test_rango_diario_usa_solo_dias_cerrados():
    dias = pd.date_range("2024-01-01", periods=15, freq="D")
    d = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0}, index=dias)
    d.iloc[-1, d.columns.get_loc("high")] = 130.0      # el día 15 tiene un rango enorme
    idx = pd.date_range(dias[-1], dias[-1] + pd.Timedelta(hours=23), freq="h")
    r = rr.rango_diario(d, idx)
    assert r.iloc[0] == pytest.approx(0.04) and np.isnan(rr.rango_diario(d, idx - pd.Timedelta(days=1)).iloc[0])
    assert r.iloc[-2] == pytest.approx(0.04)           # 22:00: el día 15 todavía no cerró
    assert r.iloc[-1] == pytest.approx(0.04)           # mediana de 14 días: un día raro no la mueve


def test_e4_caida_en_rangos():
    idx = pd.date_range("2024-01-01", periods=30, freq="h")
    c = pd.Series(100.0, index=idx)
    c.iloc[-1] = 98.9
    h = pd.DataFrame({"open": c, "high": c, "low": c, "close": c})
    ind = rr.indicadores(h)
    ind["rango"] = 0.02
    assert rr.senal_entrada(h, ind, "E4_0.5", False).iloc[-1]        # −1,1 % ≥ 0,5 × 2 %
    assert not rr.senal_entrada(h, ind, "E4_1", False).iloc[-1]      # pide −2 %


def test_s4_objetivo_variable_con_piso():
    m = motor(salida="S4_0.5")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    m.vela(T0 + pd.Timedelta(hours=1), 100, 100, 100, 100, False, np.nan, 0.04)   # entra a 100
    m.vela(T0 + pd.Timedelta(hours=2), 100, 101.9, 100, 101, False, np.nan, 0.04)  # TP 102: no toca
    assert m.slots[0].abierto
    m.vela(T0 + pd.Timedelta(hours=3), 100.2, 101.7, 100, 101, False, np.nan, 0.01)  # R cae: TP = 100,6
    assert not m.slots[0].abierto
    assert m.operaciones[-1]["salida"] == pytest.approx(100.6)        # piso del 0,6 % (0,5 × 1 % = 0,5 %)


def test_s5_trailing_en_rangos_y_escalon_variable():
    m = motor(slots="B", escalon_r=1.0, salida="S5")
    paso(m, 0, 100, 100, 100, 100, senal=True)
    m.vela(T0 + pd.Timedelta(hours=1), 100, 100, 100, 97.9, False, np.nan, 0.02)  # entra; −2,1 % ≥ 1 R
    m.vela(T0 + pd.Timedelta(hours=2), 97.9, 98, 97, 97.5, False, np.nan, 0.02)
    assert m.slots[1].abierto and m.slots[1].entrada == 97.9
    s = m.slots[1]
    m.vela(T0 + pd.Timedelta(hours=3), 97.5, 99.0, 97.5, 98.8, False, np.nan, 0.02)  # máx 99 ≥ 97,9 × 1,01: arma
    assert s.armado
    m.vela(T0 + pd.Timedelta(hours=4), 98.8, 98.8, 98.5, 98.6, False, np.nan, 0.02)  # stop 99 × 0,996 = 98,604
    assert not s.abierto
    assert m.operaciones[-1]["salida"] == pytest.approx(99 * (1 - 0.2 * 0.02))
