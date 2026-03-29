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
