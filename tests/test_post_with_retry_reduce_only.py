from __future__ import annotations

import json


def test_post_with_retry_retries_without_reduce_only_in_hedge_mode(runtime_event_spy, monkeypatch):
    import pkg.monkey_bx as mb

    calls = []
    submit_logs = []
    responses = [
        {
            "code": 109400,
            "msg": "In the Hedge mode, the 'ReduceOnly' field can not be filled.",
            "data": {},
        },
        {
            "code": 0,
            "msg": "",
            "data": {"order": {"orderId": "123456789"}},
        },
    ]

    def _fake_post_order(
        symbol,
        quantity,
        price,
        stopPrice,
        position_side,
        type,
        side,
        **kwargs,
    ):
        calls.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "stopPrice": stopPrice,
                "position_side": position_side,
                "type": type,
                "side": side,
                "kwargs": dict(kwargs),
            }
        )
        idx = len(calls) - 1
        return json.dumps(responses[idx])

    monkeypatch.setattr(mb.pkg.bingx, "post_order", _fake_post_order)
    monkeypatch.setattr(mb, "_log_order_submit_attempt", lambda **kwargs: submit_logs.append(kwargs))

    ok, details = mb._post_with_retry(
        "HBAR-USDT",
        22,
        0,
        0,
        "LONG",
        "MARKET",
        "SELL",
        delays=(0.1,),
        order_kwargs={"reduceOnly": True},
        return_details=True,
    )

    assert ok is True
    assert str(details.get("order_id")) == "123456789"
    assert len(calls) == 2
    assert calls[0]["kwargs"].get("reduceOnly") is True
    assert "reduceOnly" not in calls[1]["kwargs"]
    assert len(submit_logs) == 2

    lifecycle_categories = [x["category"] for x in runtime_event_spy["lifecycle"]]
    assert "execution_quality_warning" in lifecycle_categories
    assert "entry_order_submitted" in lifecycle_categories

    ledger_types = [x["event_type"] for x in runtime_event_spy["ledger"]]
    assert "order_submit_retry_without_reduce_only" in ledger_types
    assert "order_submitted" in ledger_types
