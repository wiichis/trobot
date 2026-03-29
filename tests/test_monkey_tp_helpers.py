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
