"""Los order_id de TP deben sobrevivir EXACTOS al ciclo guardar -> leer del CSV.

Contexto (01/10/2026). `_load_state_df` leía sin dtype: con una fila de id vacío
(tp_stage=none, lo normal) las columnas `tpN_order_id` volvían float64 y el id de
BingX salía como '2.103045465791529e+18'. `_infer_tp_idx_from_state_order_id` nunca
hacía match y el job de transiciones no atribuyó ningún fill de TP desde el 28/07
(los 41 fills vinieron de `reconcile_before_submit`; 0 `tp3_filled`).

Además, `upsert_tp_state` hacía `df.loc[len(df)] = ...` tras filtrar la fila: con el
índice agujereado eso sobrescribía la ÚLTIMA fila, borrando el estado de otro par.

Los tests de antes armaban una sola fila con ids tipo "oid-tp1" (columna object), y
por eso no veían ninguno de los dos.
"""
import pkg.tp_stage_state as tps
from tests.conftest import make_orders_df

OID = "2103045465791528960"   # TP1 LIMIT real de ONDO, 24/09


def _seed(symbol="ONDO-USDT", side="SHORT", oid=OID):
    tps.set_tp_submitted(
        symbol, side, tp_idx=1, order_id=oid, qty=29.0, price=0.4251,
        submit_position_qty=86.0, tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )


def test_el_id_vuelve_exacto_con_otra_fila_de_id_vacio(temp_tp_state):
    _seed()
    tps.upsert_tp_state("CFX-USDT", "LONG", tp_stage="none", tp_mode="partial_limit_tp")

    df = tps._load_state_df()
    fila = df[df["symbol"] == "ONDO-USDT"].iloc[0]
    assert fila["tp1_order_id"] == OID
    assert df[df["symbol"] == "CFX-USDT"].iloc[0]["tp1_order_id"] == ""


def test_el_id_no_se_degrada_tras_varios_guardados(temp_tp_state):
    """El CSV se reescribe en cada upsert; con float el id derivaba de forma permanente."""
    _seed()
    tps.upsert_tp_state("CFX-USDT", "LONG", tp_stage="none")
    for _ in range(5):
        tps.set_break_even_state("ONDO-USDT", "SHORT", "active")
        tps.set_break_even_state("CFX-USDT", "LONG", "inactive")
    assert tps.get_tp_state("ONDO-USDT", "SHORT")["tp1_order_id"] == OID
    assert OID in temp_tp_state.read_text(encoding="utf-8")


def test_un_upsert_no_borra_la_fila_de_otro_par(temp_tp_state):
    _seed("BCH-USDT", "LONG", oid="2097543930903003136")
    _seed("AVAX-USDT", "SHORT", oid="2097543930903004160")
    # BCH ya no es la última fila: antes esto sobrescribía a AVAX.
    tps.set_break_even_state("BCH-USDT", "LONG", "active")

    st = tps.get_tp_state("AVAX-USDT", "SHORT")
    assert st["tp_stage"] == "tp1_live"
    assert st["tp1_order_id"] == "2097543930903004160"
    assert len(tps._load_state_df()) == 2


def test_transiciones_atribuye_el_fill_tras_el_ciclo_csv(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb

    _seed()
    tps.upsert_tp_state("CFX-USDT", "LONG", tp_stage="none", tp_mode="partial_limit_tp")
    # Este test mide el match de ids tras el CSV; la confirmación por exchange tiene
    # sus propios tests (test_tp_confirmacion_exchange.py).
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "inferred")
    monkeypatch.setattr(mb, "total_positions",
                        lambda _s: ("ONDO-USDT", "SHORT", 0.43, 57.0, 0.0))  # 86 -> 57
    prev_df = make_orders_df([{"symbol": "ONDO-USDT", "orderId": int(OID), "type": "LIMIT",
                               "side": "BUY", "positionSide": "SHORT", "price": 0.4251}])

    mb._log_pending_order_transitions(prev_df, make_orders_df([]))

    filled = [e for e in runtime_event_spy["lifecycle"] if e["category"] == "tp1_filled"]
    assert len(filled) == 1
    assert filled[0]["order_id"] == OID
    assert filled[0]["source"] == "inferred_pending_gone_plus_position_reduction"
    assert tps.get_tp_state("ONDO-USDT", "SHORT")["tp_stage"] == "tp1_filled"
