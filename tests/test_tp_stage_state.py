from __future__ import annotations


def test_tp_stage_state_tracks_submit_baselines(temp_tp_state):
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=1,
        order_id="oid-1",
        qty=10,
        price=0.111,
        submit_position_qty=30,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )
    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=2,
        order_id="oid-2",
        qty=8,
        price=0.121,
        submit_position_qty=20,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )
    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=3,
        order_id="oid-3",
        qty=7,
        price=0.131,
        submit_position_qty=12,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )

    st = tps.get_tp_state("HBAR-USDT", "LONG")
    assert st["tp_stage"] == "tp3_live"
    assert str(st["tp1_order_id"]) == "oid-1"
    assert str(st["tp2_order_id"]) == "oid-2"
    assert str(st["tp3_order_id"]) == "oid-3"
    assert float(st["tp1_submit_position_qty"]) == 30.0
    assert float(st["tp2_submit_position_qty"]) == 20.0
    assert float(st["tp3_submit_position_qty"]) == 12.0


def test_tp_stage_fill_progression(temp_tp_state):
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted("DOGE-USDT", "SHORT", tp_idx=1, order_id="s1", qty=100, price=0.2, submit_position_qty=300)
    st1 = tps.set_tp_filled("DOGE-USDT", "SHORT", tp_idx=1)
    st2 = tps.set_tp_filled("DOGE-USDT", "SHORT", tp_idx=2)
    st3 = tps.set_tp_filled("DOGE-USDT", "SHORT", tp_idx=3)

    assert st1["tp_stage"] == "tp1_filled"
    assert st1["break_even_state"] == "pending"
    assert st2["tp_stage"] == "tp2_filled"
    assert st3["tp_stage"] == "tp3_filled"


def test_tp_stage_state_save_failure_is_non_fatal(temp_tp_state, monkeypatch):
    import pkg.tp_stage_state as tps

    def _raise_replace(*_args, **_kwargs):
        raise PermissionError("denied_for_test")

    monkeypatch.setattr(tps.os, "replace", _raise_replace)

    row = tps.upsert_tp_state(
        "XMR-USDT",
        "LONG",
        tp_mode="partial_limit_tp",
        tp_stage="tp1_live",
        break_even_state="inactive",
    )
    status = tps.get_tp_state_persist_status()

    assert row["persist_ok"] is False
    assert status["ok"] is False
    assert "denied_for_test" in str(status.get("error", ""))


def test_recolocar_el_mismo_tramo_no_degrada_la_base(temp_tp_state):
    """DYDX 26/07: el fill de TP1 no se confirmó, el bot recolocó 'tp1' y la base
    del reparto pasó de 314.7 a la posición viva, encogiendo cada tramo."""
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "DYDX-USDT", "LONG", tp_idx=1, order_id="oid-1", qty=103.8,
        price=0.12632, submit_position_qty=314.7, tp_mode="partial_limit_tp",
    )
    # Recolocación del mismo tramo con la posición ya reducida por el fill.
    st = tps.set_tp_submitted(
        "DYDX-USDT", "LONG", tp_idx=1, order_id="oid-2", qty=103.8,
        price=0.12646, submit_position_qty=210.9, tp_mode="partial_limit_tp",
    )

    assert float(st["tp1_submit_position_qty"]) == 314.7
    assert st["tp1_order_id"] == "oid-2"


def test_abrir_posicion_resetea_los_tramos_previos(temp_tp_state):
    """tp_stage='none' es el reset de plan: una posición nueva no hereda la base."""
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "DYDX-USDT", "LONG", tp_idx=1, order_id="oid-1", qty=103.8,
        price=0.12632, submit_position_qty=314.7, tp_mode="partial_limit_tp",
    )
    tps.upsert_tp_state("DYDX-USDT", "LONG", tp_stage="none", break_even_state="inactive")

    import pkg.monkey_bx as mb

    st = tps.get_tp_state("DYDX-USDT", "LONG")
    # El order_id vuelve como NaN del CSV; lo que importa es que ya no matchee.
    assert mb._norm_order_id(st.get("tp1_order_id")) == ""
    assert mb._infer_tp_idx_from_state_order_id(st, "oid-1") is None
    assert tps._safe_float_or_none(st.get("tp1_submit_position_qty")) is None

    st2 = tps.set_tp_submitted(
        "DYDX-USDT", "LONG", tp_idx=1, order_id="oid-9", qty=20.0,
        price=0.13, submit_position_qty=60.0, tp_mode="partial_limit_tp",
    )
    assert float(st2["tp1_submit_position_qty"]) == 60.0
