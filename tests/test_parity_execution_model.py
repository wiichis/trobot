"""Modelo de ejecución del parity-sim (23/09/2026).

Contexto: cruzando el parity posición por posición contra los trades reales del
25/08-20/09 aparecieron cuatro huecos, todos en la ejecución y ninguno en las señales
(41 de 43 entradas reales tenían su posición en el sim):

1. Entradas: el sim llenaba el 100% al cierre de la vela; en vivo son LIMIT PostOnly que
   expiran sin llenar ~21-26% de las veces.
2. Costos: el sim cobraba taker + slippage en entradas y TPs, que en vivo son maker
   (2 bps exactos, sin slippage). Costo por posición sim 0,073 vs real 0,032.
3. El ratchet del stop (activo desde el 05/09) no existía en el sim.
4. El `peso` por par de best_prod.json se ignoraba.

Además el live arma el BE en cuanto llena TP1 (`be_forzado_por_tp1`), aunque el cierre de
la vela haya vuelto por debajo de be_trigger.
"""
from typing import List

import numpy as np
import pandas as pd
import pytest

import pkg.backtesting as bt
import pkg.live_runtime_config as lrc
from pkg.backtesting import (
    LivePosition,
    PARITY_LEGACY_EXECUTION,
    PARITY_MAKER_FEE,
    PARITY_TAKER_FEE,
    _apply_ratchet_parity,
    _close_position_portion_live,
    _ratchet_candidate_parity,
    _update_live_sl,
    _update_ratchet_stage_parity,
)

SYM = "BCH-USDT"  # tick 0.01, qty_step 0.01 en SYMBOL_TRADING_RULES
RATCHET = {"enabled": True, "after_min": 30.0, "buffer_pct": 0.0025, "min_improve_pct": 0.0005}


def _pos(side="long", entry=100.0, sl=98.0, tp_fills=0, plan_filled=(False, False, False),
         be_trigger=0.004, qty=1.0) -> LivePosition:
    plan = [
        {"price": entry * (1.0084 if side == "long" else 0.9916), "qty": 0.33, "label": "TP1", "filled": plan_filled[0]},
        {"price": entry * (1.02 if side == "long" else 0.98), "qty": 0.33, "label": "TP2", "filled": plan_filled[1]},
        {"price": entry * (1.032 if side == "long" else 0.968), "qty": 0.34, "label": "TP3", "filled": plan_filled[2]},
    ]
    for t in plan:
        if t["filled"]:
            t["qty"] = 0.0
    return LivePosition(
        symbol=SYM, side=side, entry_time=pd.Timestamp("2026-09-01", tz="UTC"), entry_price=entry,
        qty=qty, remaining_qty=qty, entry_fee_remaining=0.0, slippage_in_remaining=0.0,
        funding_remaining=0.0, sl_price=sl, tp_plan=plan, be_trigger=be_trigger,
        be_mode="price_trigger", be_offset=0.0002, tp1_filled=False, cooldown_bars=0,
        position_id=1, tp_fills=tp_fills,
    )


# ---------------------------------------------------------------- costos


def test_tp_maker_paga_2bps_y_no_tiene_slippage():
    p = _pos()
    _, comm, slip, trade = _close_position_portion_live(
        p, 0.33, 101.0, pd.Timestamp("2026-09-01 01:00", tz="UTC"), atr_pct=0.004,
        taker_fee=PARITY_MAKER_FEE, exit_reason="TP", slippage=False)
    assert trade.slippage_out == 0.0
    assert trade.commission_out == pytest.approx(101.0 * 0.33 * 0.0002)


def test_stop_sigue_pagando_taker_con_slippage():
    p = _pos()
    _, _, _, trade = _close_position_portion_live(
        p, 1.0, 98.0, pd.Timestamp("2026-09-01 01:00", tz="UTC"), atr_pct=0.004,
        taker_fee=PARITY_TAKER_FEE, exit_reason="SL")
    assert trade.slippage_out > 0
    assert trade.commission_out == pytest.approx(98.0 * (1 - bt.calc_slippage_rate(0.004)) * 0.0005)


# ---------------------------------------------------------------- BE forzado


def test_be_se_arma_al_llenar_tp1_aunque_el_cierre_vuelva_atras():
    """TP1 lleno PRUEBA que el precio cruzó be_trigger; el cierre de la vela no importa."""
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    row = pd.Series({"SL_L": 98.0})
    _update_live_sl(p, row, price=100.1, force_be_after_tp1=True)  # 100.1 < 100.4 (be_trigger)
    assert p.sl_price == pytest.approx(100.0 * 1.0002)


def test_sin_forzar_el_be_depende_del_cierre():
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    _update_live_sl(p, pd.Series({"SL_L": 98.0}), price=100.1, force_be_after_tp1=False)
    assert p.sl_price == 98.0


def test_el_be_forzado_no_actua_sin_tp_lleno():
    p = _pos(tp_fills=0)
    _update_live_sl(p, pd.Series({"SL_L": 98.0}), price=100.1, force_be_after_tp1=True)
    assert p.sl_price == 98.0


# ---------------------------------------------------------------- ratchet


def _etapa(p, highs, lows=None, closes=None):
    lows = lows or [h - 0.3 for h in highs]
    closes = closes or [h - 0.1 for h in highs]
    for h, l, c in zip(highs, lows, closes):
        _update_ratchet_stage_parity(p, pd.Series({"high": h, "low": l}), c)


def test_ratchet_no_actua_sin_un_tramo_lleno():
    p = _pos(tp_fills=0)
    _etapa(p, [101.5] * 8)
    assert p.stage_best is None
    assert _ratchet_candidate_parity(p, RATCHET) is None


def test_ratchet_espera_after_min():
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    _etapa(p, [101.5] * 5, closes=[101.45] * 5)  # 25 min
    assert _ratchet_candidate_parity(p, RATCHET) is None
    _etapa(p, [101.5], closes=[101.45])           # 30 min
    assert _ratchet_candidate_parity(p, RATCHET) == pytest.approx(101.5 * (1 - 0.0025))


def test_ratchet_descarta_un_stop_del_lado_equivocado_del_precio():
    """El precio rebotó: el candidato quedaría por encima del precio vivo (110412)."""
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    _etapa(p, [103.0] + [101.0] * 6, closes=[102.9] + [101.0] * 6)
    # candidato 103*(1-0.0025)=102.74 > 101*(1-0.001)
    assert _ratchet_candidate_parity(p, RATCHET) is None


def test_ratchet_nunca_baja_del_break_even():
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    _etapa(p, [100.1] * 6, closes=[100.3] * 6)
    assert _ratchet_candidate_parity(p, RATCHET) == pytest.approx(100.0 * 1.0002)


def test_ratchet_no_actua_si_no_quedan_tramos():
    p = _pos(tp_fills=3, plan_filled=(True, True, True))
    _etapa(p, [104.0] * 8)
    assert _ratchet_candidate_parity(p, RATCHET) is None


def test_ratchet_exige_be_trigger_como_en_el_live():
    p = _pos(tp_fills=1, plan_filled=(True, False, False), be_trigger=0.0)
    _etapa(p, [101.5] * 8)
    assert _ratchet_candidate_parity(p, RATCHET) is None


def test_ratchet_short_es_simetrico():
    p = _pos(side="short", sl=102.0, tp_fills=1, plan_filled=(True, False, False))
    _etapa(p, [99.0] * 6, lows=[98.5] * 6, closes=[98.6] * 6)
    assert _ratchet_candidate_parity(p, RATCHET) == pytest.approx(98.5 * 1.0025)
    assert _apply_ratchet_parity(p, RATCHET) is True
    assert p.sl_price == pytest.approx(98.75)  # 98.74625 redondeado al tick HACIA ARRIBA


def test_ratchet_solo_aplica_si_mejora_lo_suficiente():
    p = _pos(tp_fills=1, plan_filled=(True, False, False), sl=101.2)
    _etapa(p, [101.5] * 6)
    # candidato 101.246 vs stop 101.2: mejora 0,045% < min_improve 0,05%
    assert _apply_ratchet_parity(p, RATCHET) is False
    assert p.sl_price == 101.2


# ---------------------------------------------------------------- integración


T0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")


def _bars(ohlc: List[tuple], signal_bars=(5,), side="long"):
    rows = []
    for i, (o, h, l, c) in enumerate(ohlc):
        rows.append({
            "symbol": SYM, "date": T0 + pd.Timedelta(minutes=5 * i),
            "open": o, "high": h, "low": l, "close": c, "volume": 1.0,
            "Long_Signal": side == "long" and i in signal_bars,
            "Short_Signal": side == "short" and i in signal_bars,
            "SL_L": 98.0, "SL_S": 102.0, "ATR_pct": 0.002,
        })
    return pd.DataFrame(rows)


def _flat(n, px=100.0):
    return [(px, px + 0.05, px - 0.05, px)] * n


@pytest.fixture
def parity(monkeypatch):
    """Parity con velas y params sintéticos y la config del LIVE fijada en el test."""
    params = {SYM: {"tp": 0.02, "tp_mode": "fixed", "sl_mode": "percent", "sl_pct": 0.02,
                    "be_trigger": 0.004, "cooldown": 0}}
    monkeypatch.setattr(lrc, "get_entry_mode", lambda: "limit_post_only")
    monkeypatch.setattr(lrc, "get_entry_limit_offset_bps", lambda: 2.0)
    monkeypatch.setattr(lrc, "get_tp_mode", lambda: "partial_limit_tp")
    monkeypatch.setattr(lrc, "get_tp_one_at_a_time", lambda: True)
    monkeypatch.setattr(lrc, "get_ratchet_config", lambda: dict(RATCHET))
    monkeypatch.setattr(lrc, "get_post_sl_cooldown_bars", lambda: 0)
    monkeypatch.setattr(lrc, "get_loss_time_stop_bars", lambda: 0)
    monkeypatch.setattr(bt, "_runtime_tp_splits_parity", lambda: (0.33, 0.33, 0.34))

    def run(df, extra_params=None, **kw):
        p = {SYM: dict(params[SYM], **(extra_params or {}))}
        monkeypatch.setattr(bt, "_load_params_map", lambda _path: p)
        monkeypatch.setattr(bt, "load_candles", lambda *_a, **_k: df.drop(
            columns=["Long_Signal", "Short_Signal", "SL_L", "SL_S", "ATR_pct"]))
        monkeypatch.setattr(bt._indicadores, "_calc_symbol",
                            lambda _d, _s, params_override=None: df.copy())
        return bt.run_live_parity_portfolio([SYM], "unused.csv", 1000.0, None,
                                            best_path="unused.json", return_trades=True, **kw)
    return run


def test_la_vela_siguiente_a_la_senal_no_puede_llenar_la_entrada(parity):
    """La orden se somete 3 min dentro de t+1: un mínimo de esa vela es anterior a ella."""
    ohlc = _flat(6) + [(100.0, 100.2, 99.0, 100.2)] + _flat(5, 100.2)
    res = parity(_bars(ohlc))
    assert res["entries_submitted"] == 1
    assert res["entries_expired"] == 1
    assert res["trades"] == 0


def test_la_entrada_llena_si_se_atraviesa_el_limite_en_su_ventana(parity):
    # señal en la vela 5 (close 100) -> límite 99.98; la vela 7 lo atraviesa
    ohlc = _flat(7) + [(100.0, 100.05, 99.9, 100.0)] + _flat(4)
    res = parity(_bars(ohlc))
    assert res["entries_expired"] == 0
    log = res["entries_log"][0]
    assert log["limit"] == pytest.approx(99.98)
    assert log["filled_time"] == T0 + pd.Timedelta(minutes=35)
    t = res["trades_list"][0]
    assert t.entry_price == pytest.approx(99.98)
    assert t.slippage_in == 0.0
    assert t.commission_in == pytest.approx(99.98 * t.qty * 0.0002)


def test_la_orden_expira_despues_de_la_vela_t_mas_4(parity):
    # el límite recién se atraviesa en la vela 10 = t+5: ya expiró
    ohlc = _flat(10, 100.2) + [(100.2, 100.2, 99.9, 100.0)] + _flat(3)
    ohlc[5] = (100.0, 100.05, 99.95, 100.0)  # vela de la señal, close 100
    res = parity(_bars(ohlc))
    assert res["entries_expired"] == 1
    assert res["trades"] == 0


def test_una_orden_pendiente_bloquea_nuevas_senales_del_par(parity):
    ohlc = _flat(6) + _flat(6, 100.2)
    res = parity(_bars(ohlc, signal_bars=(5, 6, 7)))
    assert res["entries_submitted"] == 1


def test_modo_legacy_reproduce_el_fill_instantaneo(parity):
    ohlc = _flat(6) + _flat(6, 100.2)
    res = parity(_bars(ohlc), **PARITY_LEGACY_EXECUTION)
    t = res["trades_list"][0]
    assert t.entry_time == T0 + pd.Timedelta(minutes=25)
    assert t.entry_price == pytest.approx(100.0)
    assert t.slippage_in > 0  # taker con slippage


def test_peso_por_par_se_respeta(parity):
    ohlc = _flat(7) + [(100.0, 100.05, 99.9, 100.0)] + _flat(4)
    res = parity(_bars(ohlc), extra_params={"peso": 0.13})
    # 1000 * 0.13 / 100 = 1.30 (qty_step 0.01)
    assert res["trades_list"][0].qty + sum(t.qty for t in res["trades_list"][1:]) == pytest.approx(1.30)


def test_peso_se_acota_al_piso_del_escalonado(parity, monkeypatch):
    monkeypatch.setattr(lrc, "get_tp_min_close_notional_usdt", lambda: 7.0)
    ohlc = _flat(7) + [(100.0, 100.05, 99.9, 100.0)] + _flat(4)
    res = parity(_bars(ohlc), extra_params={"peso": 0.01})
    # piso = (7 / 0.33) / 1000 = 0.0212 -> qty 0.21
    assert sum(t.qty for t in res["trades_list"]) == pytest.approx(0.21)


def _camino_con_tp1_y_retroceso():
    """Entra en 99.98 (vela 7), llena TP1 (~100.82), sube a 101.5, se queda arriba
    40 min y después cae hasta 99.9."""
    ohlc = _flat(7) + [(100.0, 100.05, 99.9, 100.0)]          # 7: fill entrada
    ohlc += [(100.3, 100.9, 100.25, 100.85)]                  # 8: TP1 atravesado
    ohlc += [(100.85, 101.5, 100.8, 101.4)]                   # 9: máximo
    ohlc += [(101.4, 101.45, 101.35, 101.4)] * 8              # 10-17: lateral arriba
    ohlc += [(101.4, 101.4, 99.9, 100.0)]                     # 18: caída
    ohlc += _flat(3, 100.0)
    return ohlc


def test_el_ratchet_asegura_parte_del_maximo(parity):
    res = parity(_bars(_camino_con_tp1_y_retroceso()))
    sl = [t for t in res["trades_list"] if t.exit_reason == "SL"]
    assert len(sl) == 1
    assert sl[0].exit_price == pytest.approx(101.24)  # 101.5 * (1 - 0.0025), al tick


def test_sin_ratchet_el_mismo_camino_sale_en_break_even(parity):
    res = parity(_bars(_camino_con_tp1_y_retroceso()), ratchet=False)
    sl = [t for t in res["trades_list"] if t.exit_reason == "SL"]
    assert len(sl) == 1
    assert sl[0].exit_price == pytest.approx(99.98 * 1.0002)  # BE forzado tras TP1


# ---------------------------------------------------------------- variantes del ratchet (24/09)


def test_parity_ratchet_por_atr_usa_el_atr_de_la_ultima_vela_cerrada():
    p = _pos(tp_fills=1, plan_filled=(True, False, False))
    for _ in range(6):
        _update_ratchet_stage_parity(p, pd.Series({"high": 101.5, "low": 101.2, "ATR_pct": 0.002}), 101.45)
    cfg = dict(RATCHET, buffer_mode="atr", buffer_atr_mult=2.0)
    assert _ratchet_candidate_parity(p, cfg) == pytest.approx(101.5 * (1 - 0.004))


def test_parity_ratchet_solo_tras_tp2():
    cfg = dict(RATCHET, only_after_tp2=True)
    p1 = _pos(tp_fills=1, plan_filled=(True, False, False))
    p2 = _pos(tp_fills=2, plan_filled=(True, True, False))
    for p in (p1, p2):
        _etapa(p, [102.5] * 6, closes=[102.45] * 6)
    assert _ratchet_candidate_parity(p1, cfg) is None
    assert _ratchet_candidate_parity(p2, cfg) == pytest.approx(102.5 * (1 - 0.0025))
