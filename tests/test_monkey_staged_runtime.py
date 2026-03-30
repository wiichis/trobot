from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

def make_orders_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["symbol", "orderId", "type", "side", "positionSide", "price", "stopPrice", "time"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def _write_runtime_files(base: Path, *, symbol: str, side: str = "LONG") -> None:
    archivos = base / "archivos"
    archivos.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": symbol, "tipo": side, "counter": 0}]).to_csv(
        archivos / "position_id_register.csv",
        index=False,
    )
    pd.DataFrame(
        columns=["symbol", "orderId", "type", "side", "positionSide", "price", "stopPrice", "time"]
    ).to_csv(archivos / "order_id_register.csv", index=False)
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "TP1_L": 101.0,
                "TP2_L": 102.0,
                "TP3_L": 103.0,
                "TP1_S": 99.0,
                "TP2_S": 98.0,
                "TP3_S": 97.0,
                "Stop_Loss_Long": 95.0,
                "Stop_Loss_Short": 105.0,
            }
        ]
    ).to_csv(archivos / "indicadores.csv", index=False)


def test_legacy_pending_gone_regression_marks_tp_fill(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=1,
        order_id="101",
        qty=10.0,
        price=0.2,
        submit_position_qty=30.0,
        tp_mode="legacy_market_tp",
        fill_confirmation_mode="inferred",
    )
    prev_df = make_orders_df(
        [
            {
                "symbol": "HBAR-USDT",
                "orderId": "101",
                "type": "TAKE_PROFIT_MARKET",
                "side": "SELL",
                "positionSide": "LONG",
                "stopPrice": 0.2,
            }
        ]
    )
    curr_df = make_orders_df([])
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 0.2, 19.0, 0.0),  # reducción suficiente vs qty TP1=10
    )
    mb._log_pending_order_transitions(prev_df, curr_df)

    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "take_profit_hit" in categories
    assert "tp1_filled" in categories
    st = tps.get_tp_state("HBAR-USDT", "LONG")
    assert st["tp_stage"] == "tp1_filled"


def test_partial_mode_submits_tp2_only_after_tp1_filled(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    symbol = "HBAR-USDT"
    _write_runtime_files(isolated_workspace, symbol=symbol, side="LONG")
    tps.upsert_tp_state(
        symbol,
        "LONG",
        tp_mode="partial_limit_tp",
        tp_stage="tp1_filled",
        tp_fill_confirmation_mode="inferred",
        tp1_submit_position_qty=30.0,
    )

    monkeypatch.setattr(mb, "get_tp_mode", lambda: "partial_limit_tp")
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "inferred")
    monkeypatch.setattr(mb, "get_tp_limit_offset_bps", lambda: 0.0)
    monkeypatch.setattr(mb, "get_tp_legacy_fallback_on_error", lambda: True)
    monkeypatch.setattr(mb, "obteniendo_ordenes_pendientes", lambda: [])
    monkeypatch.setattr(mb, "_last_traded_price", lambda _symbol: 100.0)
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: (symbol, "LONG", 100.0, 15.0, 0.0),
    )

    calls = []

    def _post(symbol, qty, price, stop, position_side, order_type, side, **kwargs):
        calls.append(
            {
                "symbol": symbol,
                "qty": float(qty),
                "price": float(price),
                "stop": float(stop),
                "position_side": position_side,
                "order_type": order_type,
                "side": side,
                "kwargs": kwargs,
            }
        )
        if kwargs.get("return_details"):
            return True, {"order_id": f"oid-{len(calls)}", "request_id": "r", "submit_time_utc": "2026-01-01T00:00:00Z"}
        return True

    monkeypatch.setattr(mb, "_post_with_retry", _post)
    mb.colocando_TK_SL()

    tp_limit_calls = [c for c in calls if c["order_type"] == "LIMIT" and c["side"] == "SELL"]
    assert len(tp_limit_calls) == 1
    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "tp2_submitted" in categories
    assert "tp1_submitted" not in categories


def test_partial_mode_submits_tp3_only_after_tp2_filled_short(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    symbol = "DOGE-USDT"
    _write_runtime_files(isolated_workspace, symbol=symbol, side="SHORT")
    tps.upsert_tp_state(
        symbol,
        "SHORT",
        tp_mode="partial_limit_tp",
        tp_stage="tp2_filled",
        tp_fill_confirmation_mode="inferred",
        tp1_submit_position_qty=24.0,
    )

    monkeypatch.setattr(mb, "get_tp_mode", lambda: "partial_limit_tp")
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "inferred")
    monkeypatch.setattr(mb, "get_tp_limit_offset_bps", lambda: 0.0)
    monkeypatch.setattr(mb, "get_tp_legacy_fallback_on_error", lambda: True)
    monkeypatch.setattr(mb, "obteniendo_ordenes_pendientes", lambda: [])
    monkeypatch.setattr(mb, "_last_traded_price", lambda _symbol: 100.0)
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: (symbol, "SHORT", 100.0, 4.8, 0.0),
    )

    calls = []

    def _post(symbol, qty, price, stop, position_side, order_type, side, **kwargs):
        calls.append((order_type, side, float(qty), float(price), float(stop), kwargs))
        if kwargs.get("return_details"):
            return True, {"order_id": f"oid-{len(calls)}", "request_id": "r", "submit_time_utc": "2026-01-01T00:00:00Z"}
        return True

    monkeypatch.setattr(mb, "_post_with_retry", _post)
    mb.colocando_TK_SL()

    tp_limit_calls = [c for c in calls if c[0] == "LIMIT" and c[1] == "BUY"]
    assert len(tp_limit_calls) == 1
    assert tp_limit_calls[0][2] <= 4.8  # no over-close en SHORT
    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "tp3_submitted" in categories
    assert "tp2_submitted" not in categories


def test_stage_confirmation_failure_uses_fallback_when_enabled(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=2,
        order_id="202",
        qty=2.0,
        price=101.0,
        submit_position_qty=10.0,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )
    prev_df = make_orders_df(
        [
            {
                "symbol": "HBAR-USDT",
                "orderId": "202",
                "type": "LIMIT",
                "side": "SELL",
                "positionSide": "LONG",
                "price": 101.0,
            }
        ]
    )
    curr_df = make_orders_df([])

    monkeypatch.setattr(mb, "get_tp_legacy_fallback_on_error", lambda: True)
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 100.0, 9.7, 0.0),  # reduccion insuficiente
    )
    fb_calls = []
    monkeypatch.setattr(mb, "_submit_tp_legacy_fallback", lambda **kwargs: fb_calls.append(kwargs) or True)

    mb._log_pending_order_transitions(prev_df, curr_df)
    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "tp2_failed" in categories
    assert len(fb_calls) == 1


def test_stage_confirmation_failure_no_fallback_when_disabled(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "SHORT",
        tp_idx=3,
        order_id="303",
        qty=1.0,
        price=95.0,
        submit_position_qty=4.0,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )
    prev_df = make_orders_df(
        [
            {
                "symbol": "HBAR-USDT",
                "orderId": "303",
                "type": "LIMIT",
                "side": "BUY",
                "positionSide": "SHORT",
                "price": 95.0,
            }
        ]
    )
    curr_df = make_orders_df([])

    monkeypatch.setattr(mb, "get_tp_legacy_fallback_on_error", lambda: False)
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "SHORT", 100.0, 3.8, 0.0),  # reduccion insuficiente
    )
    fb_calls = []
    monkeypatch.setattr(mb, "_submit_tp_legacy_fallback", lambda **kwargs: fb_calls.append(kwargs) or True)

    mb._log_pending_order_transitions(prev_df, curr_df)
    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "tp3_failed" in categories
    assert fb_calls == []


def test_pending_transition_idempotency_no_duplicate_tp_fill(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=1,
        order_id="111",
        qty=2.0,
        price=101.0,
        submit_position_qty=10.0,
        tp_mode="partial_limit_tp",
        fill_confirmation_mode="inferred",
    )
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 100.0, 7.5, 0.0),
    )

    prev_df = make_orders_df(
        [
            {
                "symbol": "HBAR-USDT",
                "orderId": "111",
                "type": "LIMIT",
                "side": "SELL",
                "positionSide": "LONG",
                "price": 101.0,
            }
        ]
    )
    empty_df = make_orders_df([])
    mb._log_pending_order_transitions(prev_df, empty_df)
    mb._log_pending_order_transitions(empty_df, empty_df)

    filled = [x for x in runtime_event_spy["lifecycle"] if x["category"] == "tp1_filled"]
    assert len(filled) == 1


def test_market_tp_pending_gone_requires_position_reduction_confirmation(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    tps.set_tp_submitted(
        "HBAR-USDT",
        "LONG",
        tp_idx=1,
        order_id="501",
        qty=2.0,
        price=0.21,
        submit_position_qty=10.0,
        tp_mode="legacy_market_tp",
        fill_confirmation_mode="inferred",
    )
    prev_df = make_orders_df(
        [
            {
                "symbol": "HBAR-USDT",
                "orderId": "501",
                "type": "TAKE_PROFIT_MARKET",
                "side": "SELL",
                "positionSide": "LONG",
                "stopPrice": 0.21,
            }
        ]
    )
    curr_df = make_orders_df([])
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: ("HBAR-USDT", "LONG", 0.2, 9.4, 0.0),  # reducción insuficiente
    )

    mb._log_pending_order_transitions(prev_df, curr_df)

    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "take_profit_hit" not in categories
    assert "tp1_filled" not in categories
    assert "execution_quality_warning" in categories

    event_types = [x["event_type"] for x in runtime_event_spy["ledger"]]
    assert "tp_confirmation_failed" in event_types


def test_break_even_not_activated_before_tp1_confirmation(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb
    import pkg.tp_stage_state as tps

    symbol = "HBAR-USDT"
    (Path("archivos") / "indicadores.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"symbol": symbol, "close": 102.0, "Stop_Loss_Long": 90.0, "Stop_Loss_Short": 110.0}]).to_csv(
        "archivos/indicadores.csv",
        index=False,
    )
    pd.DataFrame(columns=["symbol", "orderId", "type", "stopPrice", "time"]).to_csv(
        "archivos/order_id_register.csv",
        index=False,
    )
    best_path = Path("best_prod_test.json")
    best_path.write_text(json.dumps([{"symbol": symbol, "params": {"be_trigger": 0.01}}]), encoding="utf-8")

    tps.upsert_tp_state(symbol, "LONG", tp_mode="partial_limit_tp", tp_stage="tp1_live")
    monkeypatch.setattr(mb, "BEST_PROD_PATH", str(best_path))
    monkeypatch.setattr(mb, "filtrando_posiciones_antiguas", lambda: pd.DataFrame([{"symbol": symbol, "orderId": "sl-1", "stopPrice": 100.0}]))
    monkeypatch.setattr(mb, "total_positions", lambda _symbol: (symbol, "LONG", 100.0, 10.0, 0.0))
    monkeypatch.setattr(mb, "get_tp_mode", lambda: "partial_limit_tp")
    monkeypatch.setattr(mb, "is_break_even_after_tp1_enabled", lambda: True)
    monkeypatch.setattr(mb.pkg.bingx, "cancel_order", lambda *_a, **_k: None)
    monkeypatch.setattr(mb, "obteniendo_ordenes_pendientes", lambda: [])
    monkeypatch.setattr(mb, "_post_with_retry", lambda *_a, **_k: True)

    mb.unrealized_profit_positions()
    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "break_even_activated" not in categories


def test_colocando_tk_sl_forces_legacy_tp_when_tp_state_unhealthy(
    isolated_workspace, runtime_event_spy, temp_tp_state, monkeypatch
):
    import pkg.monkey_bx as mb

    symbol = "DOT-USDT"
    _write_runtime_files(isolated_workspace, symbol=symbol, side="LONG")

    monkeypatch.setattr(mb, "get_tp_mode", lambda: "partial_limit_tp")
    monkeypatch.setattr(mb, "get_tp_fill_confirmation_mode", lambda: "inferred")
    monkeypatch.setattr(mb, "get_tp_state_persist_status", lambda: {"ok": False, "error": "permission_denied"})
    monkeypatch.setattr(mb, "upsert_tp_state", lambda *_a, **_k: {"persist_ok": False})
    monkeypatch.setattr(mb, "obteniendo_ordenes_pendientes", lambda: [])
    monkeypatch.setattr(mb, "_last_traded_price", lambda _symbol: 100.0)
    monkeypatch.setattr(
        mb,
        "total_positions",
        lambda _symbol: (symbol, "LONG", 100.0, 15.0, 0.0),
    )

    calls = []

    def _post(symbol, qty, price, stop, position_side, order_type, side, **kwargs):
        calls.append(
            {
                "symbol": symbol,
                "qty": float(qty),
                "price": float(price),
                "stop": float(stop),
                "position_side": position_side,
                "order_type": order_type,
                "side": side,
                "kwargs": kwargs,
            }
        )
        if kwargs.get("return_details"):
            return True, {"order_id": f"oid-{len(calls)}", "request_id": "r", "submit_time_utc": "2026-01-01T00:00:00Z"}
        return True

    monkeypatch.setattr(mb, "_post_with_retry", _post)
    mb.colocando_TK_SL()

    order_types = [c["order_type"] for c in calls]
    assert "STOP_MARKET" in order_types
    assert "TAKE_PROFIT_MARKET" in order_types
    assert "LIMIT" not in order_types

    categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "runtime_storage_warning" in categories
    assert "legacy_fallback_used" in categories
