from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("none", 1),
        ("tp1_live", 1),
        ("tp1_filled", 2),
        ("tp2_live", 2),
        ("tp2_filled", 3),
        ("tp3_live", 3),
        ("tp3_filled", None),
    ],
)
def test_next_tp_idx_from_stage(stage, expected):
    import pkg.monkey_bx as mb

    assert mb._next_tp_idx_from_stage(stage) == expected


def test_compute_stage_qty_never_exceeds_residual():
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=1.2,
        step_sz=0.1,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 9.0},
        stage_idx=2,
    )

    assert reason == "ok"
    assert qty > 0.0
    assert qty <= 1.2  # nunca debe sobrepasar el remanente


def test_compute_stage3_qty_uses_final_residual():
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=0.257,
        step_sz=0.01,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 2.0},
        stage_idx=3,
    )

    assert reason == "ok"
    assert qty == 0.25


def test_compute_small_residual_rounding_safety():
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=0.0049,
        step_sz=0.01,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 0.03},
        stage_idx=2,
    )

    assert qty == 0.0
    assert reason in ("position_qty_now_zero", "stage_qty_rounded_zero")


def test_infer_tp_fill_from_position_confirmed(monkeypatch):
    import pkg.monkey_bx as mb

    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 0.1, 7.8, 0.0),
    )
    ok, reason, current_qty, reduction = mb._infer_tp_fill_from_position(
        "HBAR-USDT",
        "LONG",
        {"tp2_submit_position_qty": 10.0, "tp2_qty": 2.0},
        tp_idx=2,
    )

    assert ok is True
    assert "inferred_pending_gone_plus_position_reduction" in reason
    assert current_qty == 7.8
    assert reduction == pytest.approx(2.2, rel=1e-9)


def test_infer_tp_fill_from_position_not_confirmed(monkeypatch):
    import pkg.monkey_bx as mb

    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 0.1, 9.4, 0.0),
    )
    ok, reason, _current_qty, _reduction = mb._infer_tp_fill_from_position(
        "HBAR-USDT",
        "LONG",
        {"tp1_submit_position_qty": 10.0, "tp1_qty": 2.0},
        tp_idx=1,
    )

    assert ok is False
    assert reason.startswith("reduction_too_low:")


def test_sanitize_entry_limit_price_stays_maker_side():
    import pkg.monkey_bx as mb

    px_long = mb._sanitize_entry_limit_price(100.0, "BNB-USDT", "LONG", offset_bps=2.0)
    px_short = mb._sanitize_entry_limit_price(100.0, "BNB-USDT", "SHORT", offset_bps=2.0)

    assert px_long < 100.0
    assert px_short > 100.0


@pytest.mark.parametrize(
    "qty,step,expected",
    [
        (107.1, 0.1, 107.1),
        (314.7, 0.1, 314.7),
        (0.3, 0.1, 0.3),
        (0.1265, 0.00001, 0.1265),
        (6.089, 0.001, 6.089),
        (1.05, 0.1, 1.0),
        (0.07, 0.1, 0.0),
        (843.0, 1.0, 843.0),
    ],
)
def test_round_step_no_pierde_un_step_por_binario(qty, step, expected):
    """107.1/0.1 == 1070.9999... en binario: el floor se comía un step entero."""
    import pkg.monkey_bx as mb

    assert mb._round_step(qty, step) == pytest.approx(expected, abs=1e-9)


def test_stage_qty_colapsa_cuando_el_remanente_no_alcanza_el_minimo():
    """DYDX 26/07: el 3er tramo dejaba 37.6 u (~4.8 USDT) que ningún TP podía cerrar."""
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=107.1,
        step_sz=0.1,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 314.7},
        stage_idx=1,
        price_ref=0.1265,
        min_close_notional=7.0,
    )

    assert qty == pytest.approx(107.1, abs=1e-9)
    assert reason == "ok_collapsed_min_leftover"


def test_stage_qty_no_colapsa_si_el_remanente_es_operable():
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=314.7,
        step_sz=0.1,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 314.7},
        stage_idx=1,
        price_ref=0.1265,
        min_close_notional=7.0,
    )

    assert qty == pytest.approx(103.8, abs=1e-9)
    assert reason == "ok"


def test_stage_qty_sin_orden_si_ni_el_total_alcanza_el_minimo():
    """Límite del exchange, no del bot: el SL queda como única protección."""
    import pkg.monkey_bx as mb

    qty, reason = mb._compute_partial_limit_stage_qty(
        position_qty_now=40.0,
        step_sz=0.1,
        splits=(0.33, 0.33, 0.34),
        state={"tp1_submit_position_qty": 40.0},
        stage_idx=1,
        price_ref=0.1265,
        min_close_notional=7.0,
    )

    assert qty == 0.0
    assert reason == "below_min_close_notional"
