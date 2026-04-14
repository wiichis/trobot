import pkg
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
import time
import os
import uuid
from .settings import BEST_PROD_PATH

import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP
# --- Cooldown por SL: lectura de estado compartido con indicadores ---
from .live_runtime_config import (
    get_cooldown_minutes_override,
    is_entry_hour_allowed_utc,
    get_entry_mode,
    get_entry_limit_offset_bps,
    get_entry_time_in_force,
    is_entry_post_only_enabled,
    is_entry_market_fallback_on_error_enabled,
    get_tp_mode,
    get_tp_partial_distribution,
    get_tp_fill_confirmation_mode,
    get_tp_limit_offset_bps,
    get_tp_reduce_only,
    get_tp_legacy_fallback_on_error,
    is_break_even_after_tp1_enabled,
)
from .lifecycle_events import emit_lifecycle_event
from .execution_ledger import append_execution_ledger_event
from .tp_stage_state import (
    get_tp_state,
    get_tp_state_persist_status,
    upsert_tp_state,
    set_tp_submitted,
    set_tp_filled,
    set_break_even_state,
    set_sl_guard,
    is_sl_guard_active,
    clear_tp_state,
)

COOLDOWN_CSV = './archivos/cooldown.csv'

# Registro local de SL colocados para detectar fills
SL_WATCH_CSV = './archivos/sl_watch.csv'
ORDER_SUBMIT_LOG_CSV = './archivos/order_submit_log.csv'
ORDER_LIFECYCLE_LOG_CSV = './archivos/order_lifecycle_log.csv'
ORDER_PENDING_PREV_CSV = './archivos/order_pending_prev.csv'
ENTRY_WATCH_CSV = './archivos/entry_watch.csv'
RUNTIME_STORAGE_ALERT_COOLDOWN_SEC = 300
_RUNTIME_STORAGE_ALERT_LAST_TS = {}


def _norm_order_id(value) -> str:
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none'):
        return ''
    if text.endswith('.0'):
        base = text[:-2]
        if base.isdigit():
            return base
    return text


def _emit_runtime_storage_warning(
    *,
    symbol: str,
    position_side: str,
    source: str,
    reason: str,
    detail: str,
    severity: str = "CRITICAL",
    cooldown_sec: int = RUNTIME_STORAGE_ALERT_COOLDOWN_SEC,
) -> bool:
    key = "|".join(
        [
            str(source or "").strip(),
            str(symbol or "").upper().strip(),
            str(position_side or "").upper().strip(),
            str(reason or "").strip(),
        ]
    )
    now = time.time()
    last = _RUNTIME_STORAGE_ALERT_LAST_TS.get(key, 0.0)
    if (now - float(last)) < max(1, int(cooldown_sec)):
        return False
    _RUNTIME_STORAGE_ALERT_LAST_TS[key] = now

    symbol_u = str(symbol or "").upper().strip()
    pside_u = str(position_side or "").upper().strip()
    detail_txt = str(detail or "")[:220]
    emit_lifecycle_event(
        "runtime_storage_warning",
        severity,
        symbol=symbol_u,
        position_side=pside_u,
        reason=str(reason or "").strip(),
        detail=detail_txt,
        source=str(source or "").strip(),
    )
    append_execution_ledger_event(
        "tp_state_persist_error",
        data_quality="actual",
        source=str(source or "").strip(),
        symbol=symbol_u,
        position_side=pside_u,
        cancel_reason=str(reason or "").strip(),
        raw_msg=detail_txt,
    )
    return True


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + 'Z'


def _append_csv_row(path: str, row: dict, columns: list) -> None:
    try:
        df_new = pd.DataFrame([row], columns=columns)
        exists = os.path.exists(path) and os.path.getsize(path) > 0
        df_new.to_csv(path, mode='a' if exists else 'w', header=not exists, index=False)
    except Exception as e:
        print(f"Error escribiendo log CSV {path}: {e}")


def _extract_order_id_from_payload(payload):
    try:
        if isinstance(payload, dict):
            for key in ('orderId', 'order_id', 'id'):
                if key in payload:
                    oid = _norm_order_id(payload.get(key))
                    if oid:
                        return oid
            for val in payload.values():
                oid = _extract_order_id_from_payload(val)
                if oid:
                    return oid
        elif isinstance(payload, list):
            for item in payload:
                oid = _extract_order_id_from_payload(item)
                if oid:
                    return oid
    except Exception:
        return None
    return None


def _log_order_submit_attempt(
    *,
    request_id: str,
    symbol: str,
    qty,
    price,
    stop_price,
    position_side: str,
    order_type: str,
    side: str,
    attempt: int,
    delay_s: float,
    accepted: bool,
    code,
    msg: str,
    order_id: str,
    raw_response: str,
) -> None:
    cols = [
        'ts_utc', 'request_id', 'symbol', 'order_type', 'position_side', 'side',
        'qty', 'price', 'stop_price', 'attempt', 'delay_s',
        'accepted', 'code', 'msg', 'order_id', 'raw_response',
    ]
    row = {
        'ts_utc': _utc_now_iso(),
        'request_id': str(request_id),
        'symbol': str(symbol).upper(),
        'order_type': str(order_type).upper(),
        'position_side': str(position_side).upper(),
        'side': str(side).upper(),
        'qty': _safe_float(qty) if 'qty' in locals() else qty,
        'price': _safe_float(price) if 'price' in locals() else price,
        'stop_price': _safe_float(stop_price) if 'stop_price' in locals() else stop_price,
        'attempt': int(attempt),
        'delay_s': _safe_float(delay_s),
        'accepted': bool(accepted),
        'code': '' if code is None else str(code),
        'msg': str(msg)[:240] if msg is not None else '',
        'order_id': _norm_order_id(order_id),
        'raw_response': (str(raw_response)[:240] if raw_response is not None else ''),
    }
    _append_csv_row(ORDER_SUBMIT_LOG_CSV, row, cols)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_float_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text == '' or text.lower() in ('none', 'nan', 'na'):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _position_close_side(position_side: str) -> str:
    pside = str(position_side or "").upper().strip()
    return "SELL" if pside == "LONG" else "BUY"


def _normalize_order_side(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "side" not in out.columns:
        out["side"] = ""
    if "positionSide" not in out.columns:
        out["positionSide"] = ""
    out["side"] = out["side"].astype(str).str.upper().str.strip()
    out["positionSide"] = out["positionSide"].astype(str).str.upper().str.strip()
    return out


def _extract_tp_orders(symbol_orders: pd.DataFrame, position_side: str, tp_mode: str) -> pd.DataFrame:
    if symbol_orders is None or symbol_orders.empty:
        return pd.DataFrame(columns=getattr(symbol_orders, "columns", []))
    df = symbol_orders.copy()
    if "type" not in df.columns:
        df["type"] = ""
    df = _normalize_order_side(df)
    df["type"] = df["type"].astype(str).str.upper().str.strip()

    if str(tp_mode or "").lower() != "partial_limit_tp":
        return df[df["type"] == "TAKE_PROFIT_MARKET"].copy()

    close_side = _position_close_side(position_side)
    mask_market = df["type"] == "TAKE_PROFIT_MARKET"
    mask_limit = (df["type"] == "LIMIT") & (df["side"] == close_side)
    return df[mask_market | mask_limit].copy()


def _extract_tp_price_set(tp_orders: pd.DataFrame, symbol: str) -> set:
    if tp_orders is None or tp_orders.empty:
        return set()
    tick = _tick_size_for(symbol)
    out = set()
    for _, row in tp_orders.iterrows():
        otype = str(row.get("type", "")).upper().strip()
        raw = row.get("price") if otype == "LIMIT" else row.get("stopPrice")
        try:
            val = float(raw)
        except Exception:
            continue
        if val <= 0:
            continue
        out.add(_round_to_tick(val, tick))
    return out


def _sanitize_tp_limit_price(
    target_price: float,
    symbol: str,
    position_side: str,
    ref_price,
    *,
    offset_bps: float = 0.0,
) -> float:
    side = str(position_side or "").upper().strip()
    tick = _tick_size_for(symbol)
    min_px = tick if tick and tick > 0 else 1e-8
    px = float(target_price)
    bps = max(0.0, _safe_float(offset_bps, 0.0))
    if bps > 0:
        mult = 1.0 - (bps / 10000.0) if side == "LONG" else 1.0 + (bps / 10000.0)
        px *= max(mult, 1e-6)
    rounding = ROUND_DOWN if side == "LONG" else ROUND_UP
    px = _round_to_tick(px, tick, rounding=rounding)

    try:
        ref = float(ref_price)
    except Exception:
        ref = None
    if ref is not None and ref > 0:
        if side == "LONG" and px <= ref:
            px = _round_to_tick(ref + tick, tick, rounding=ROUND_UP)
        if side == "SHORT" and px >= ref:
            px = _round_to_tick(max(ref - tick, ref * 0.999), tick, rounding=ROUND_DOWN)
    if px <= 0:
        px = min_px
    return float(px)


def _sanitize_entry_limit_price(
    ref_price: float,
    symbol: str,
    position_side: str,
    *,
    offset_bps: float = 2.0,
) -> float:
    """Calcula precio LIMIT para entrada maker (post-only) sin cruzar el book."""
    side = str(position_side or "").upper().strip()
    tick = _tick_size_for(symbol)
    min_px = tick if tick and tick > 0 else 1e-8
    ref = max(float(ref_price), min_px)
    bps = max(0.0, _safe_float(offset_bps, 2.0))

    if side == "LONG":
        raw = ref * (1.0 - (bps / 10000.0))
        px = _round_to_tick(raw, tick, rounding=ROUND_DOWN)
        if px >= ref:
            px = _round_to_tick(max(ref - tick, ref * 0.999), tick, rounding=ROUND_DOWN)
    else:
        raw = ref * (1.0 + (bps / 10000.0))
        px = _round_to_tick(raw, tick, rounding=ROUND_UP)
        if px <= ref:
            px = _round_to_tick(ref + tick, tick, rounding=ROUND_UP)

    if px <= 0:
        px = min_px
    return float(px)


def _sanitize_tp1_limit_price(
    target_price: float,
    symbol: str,
    position_side: str,
    ref_price,
    *,
    offset_bps: float = 0.0,
) -> float:
    """Compat alias Patch 5B."""
    return _sanitize_tp_limit_price(
        target_price,
        symbol,
        position_side,
        ref_price,
        offset_bps=offset_bps,
    )


def _tp_limit_order_kwargs(symbol: str, position_side: str, tp_idx: int) -> dict:
    idx = max(1, min(3, int(tp_idx)))
    sym = str(symbol or "").upper().replace("-", "")[:10]
    p = str(position_side or "").upper().strip()
    short_side = "L" if p == "LONG" else "S"
    client_id = f"tp{idx}{sym}{short_side}{uuid.uuid4().hex[:12]}"
    return {
        "reduceOnly": bool(get_tp_reduce_only()),
        "timeInForce": "GTC",
        "clientOrderId": client_id,
    }


def _tp1_limit_order_kwargs(symbol: str, position_side: str) -> dict:
    """Compat alias Patch 5B."""
    return _tp_limit_order_kwargs(symbol, position_side, tp_idx=1)


def _tp_stage_name(tp_idx: int) -> str:
    idx = max(1, min(3, int(tp_idx)))
    return f"tp{idx}"


def _next_tp_idx_from_stage(tp_stage: str):
    stage = str(tp_stage or "").lower().strip()
    if stage in ("", "none", "tp1_live"):
        return 1
    if stage in ("tp1_filled", "tp2_live"):
        return 2
    if stage in ("tp2_filled", "tp3_live"):
        return 3
    if stage == "tp3_filled":
        return None
    return 1


def _stage_order_id_from_state(state: dict, tp_idx: int) -> str:
    idx = max(1, min(3, int(tp_idx)))
    return _norm_order_id(state.get(f"tp{idx}_order_id"))


def _infer_tp_idx_from_state_order_id(state: dict, order_id: str):
    oid = _norm_order_id(order_id)
    if not oid:
        return None
    for idx in (1, 2, 3):
        if oid == _stage_order_id_from_state(state, idx):
            return idx
    return None


def _compute_partial_limit_stage_qty(
    *,
    position_qty_now: float,
    step_sz: float,
    splits: tuple,
    state: dict,
    stage_idx: int,
):
    idx = max(1, min(3, int(stage_idx)))
    qty_now = max(0.0, _safe_float(position_qty_now, 0.0))
    qty_now = _round_step(qty_now, step_sz)
    if qty_now <= 0:
        return 0.0, "position_qty_now_zero"

    base_qty = _safe_float_or_none(state.get("tp1_submit_position_qty"))
    if base_qty is None or base_qty <= 0:
        base_qty = qty_now
    split_qtys = _split_position_qtys(base_qty, splits, step_sz)
    if not split_qtys:
        return 0.0, "split_qty_empty"

    if idx >= 3:
        qty = qty_now
    else:
        stage_target = 0.0
        try:
            stage_target = float(split_qtys[idx - 1])
        except Exception:
            stage_target = 0.0
        qty = min(max(0.0, stage_target), qty_now)
    qty = _round_step(qty, step_sz)
    if qty <= 0:
        return 0.0, "stage_qty_rounded_zero"
    return qty, "ok"


def _emit_tp_failed(
    *,
    tp_idx: int,
    symbol: str,
    position_side: str,
    reason: str,
    detail: str = "",
    order_id: str = "",
    data_quality: str = "actual",
    source: str = "",
    tp_price=None,
    tp_qty=None,
) -> None:
    stage = _tp_stage_name(tp_idx)
    emit_lifecycle_event(
        f"{stage}_failed",
        "WARN",
        symbol=str(symbol).upper(),
        position_side=str(position_side).upper(),
        reason=str(reason),
        detail=str(detail)[:220],
        order_id=_norm_order_id(order_id),
        source=str(source or ""),
    )
    append_execution_ledger_event(
        f"{stage}_failed",
        data_quality=str(data_quality or "inferred"),
        source=str(source or ""),
        symbol=str(symbol).upper(),
        position_side=str(position_side).upper(),
        order_id=_norm_order_id(order_id),
        order_type="LIMIT",
        submitted_price=_safe_float_or_none(tp_price),
        submit_qty=_safe_float_or_none(tp_qty),
        cancel_reason=str(reason or ""),
        raw_msg=str(detail)[:240],
    )


def _emit_tp1_failed(**kwargs):
    """Compat wrapper Patch 5B."""
    _emit_tp_failed(tp_idx=1, **kwargs)


def _infer_tp_fill_from_position(symbol: str, position_side: str, state: dict, tp_idx: int):
    idx = max(1, min(3, int(tp_idx)))
    submit_qty = _safe_float_or_none(state.get(f"tp{idx}_submit_position_qty"))
    stage_qty = _safe_float_or_none(state.get(f"tp{idx}_qty"))
    if submit_qty is None or submit_qty <= 0 or stage_qty is None or stage_qty <= 0:
        return False, "baseline_qty_unavailable", None, None

    try:
        _, pside_now, _price, position_amt, _upnl = total_positions(symbol)
        if pside_now is None:
            return False, "position_unavailable", None, None
        if str(pside_now).upper().strip() != str(position_side).upper().strip():
            return False, "position_side_mismatch", None, None
        current_qty = abs(float(position_amt))
    except Exception as exc:
        return False, f"position_read_error:{exc}", None, None

    reduction = max(0.0, float(submit_qty) - float(current_qty))
    min_expected = max(float(stage_qty) * 0.60, _step_size_for(symbol) * 0.5)
    if reduction >= min_expected:
        return True, "inferred_pending_gone_plus_position_reduction", current_qty, reduction
    return False, f"reduction_too_low:{reduction:.6f}<{min_expected:.6f}", current_qty, reduction


def _infer_tp1_fill_from_position(symbol: str, position_side: str, state: dict):
    """Compat wrapper Patch 5B."""
    return _infer_tp_fill_from_position(symbol, position_side, state, tp_idx=1)


def _submit_tp_legacy_fallback(
    *,
    tp_idx: int,
    symbol: str,
    position_side: str,
    market_ref,
    tp_price,
    tp_qty,
    tp_fill_mode: str,
    reason: str,
) -> bool:
    idx = max(1, min(3, int(tp_idx)))
    tp_qty_f = _safe_float_or_none(tp_qty)
    tp_px_f = _safe_float_or_none(tp_price)
    if tp_qty_f is None or tp_qty_f <= 0 or tp_px_f is None or tp_px_f <= 0:
        return False

    side = _position_close_side(position_side)
    tp_px = _sanitize_trigger_price(float(tp_px_f), symbol, position_side, "TAKE_PROFIT_MARKET", market_ref)
    ok, details = _post_with_retry(
        symbol,
        tp_qty_f,
        0,
        tp_px,
        position_side,
        "TAKE_PROFIT_MARKET",
        side,
        return_details=True,
    )
    if not ok:
        return False

    set_tp_submitted(
        symbol,
        position_side,
        tp_idx=idx,
        order_id=details.get("order_id", ""),
        qty=tp_qty_f,
        price=tp_px,
        tp_mode="legacy_market_tp",
        fill_confirmation_mode=tp_fill_mode,
    )
    emit_lifecycle_event(
        "legacy_fallback_used",
        "WARN",
        symbol=str(symbol).upper(),
        position_side=str(position_side).upper(),
        order_id=details.get("order_id", ""),
        tp_price=tp_px,
        qty=tp_qty_f,
        tp_stage=_tp_stage_name(idx),
        reason=str(reason),
        source=f"{_tp_stage_name(idx)}_fallback_legacy_market",
    )
    append_execution_ledger_event(
        "legacy_fallback_used",
        data_quality="actual",
        source=f"{_tp_stage_name(idx)}_fallback_legacy_market",
        order_id=details.get("order_id", ""),
        request_id=details.get("request_id", ""),
        symbol=str(symbol).upper(),
        side=side,
        position_side=str(position_side).upper(),
        order_type="TAKE_PROFIT_MARKET",
        stop_price=tp_px,
        submit_qty=tp_qty_f,
        submit_time_utc=details.get("submit_time_utc", ""),
        close_reason=f"{_tp_stage_name(idx)}_fallback",
        notes=str(reason),
    )
    return True


def _submit_tp1_legacy_fallback(**kwargs) -> bool:
    """Compat wrapper Patch 5B."""
    return _submit_tp_legacy_fallback(tp_idx=1, **kwargs)


def _pending_order_key(row: dict) -> str:
    order_id = _norm_order_id(row.get('orderId'))
    if order_id:
        return f"id:{order_id}"
    symbol = str(row.get('symbol', '')).upper()
    otype = str(row.get('type', '')).upper()
    side = str(row.get('side', '')).upper()
    pside = str(row.get('positionSide', '')).upper()
    stop = row.get('stopPrice')
    try:
        stop = f"{float(stop):.8f}"
    except Exception:
        stop = ''
    return f"fallback:{symbol}|{otype}|{side}|{pside}|{stop}"


def _log_pending_order_transitions(prev_df: pd.DataFrame, curr_df: pd.DataFrame) -> None:
    cols = ['ts_utc', 'event_type', 'symbol', 'order_id', 'type', 'side', 'position_side', 'stop_price', 'source']
    ts = _utc_now_iso()

    def _rows_map(df: pd.DataFrame) -> dict:
        out = {}
        if df is None or df.empty:
            return out
        for _, r in df.iterrows():
            row = {
                'symbol': str(r.get('symbol', '')).upper(),
                'orderId': _norm_order_id(r.get('orderId')),
                'type': str(r.get('type', '')).upper(),
                'side': str(r.get('side', '')).upper(),
                'positionSide': str(r.get('positionSide', '')).upper(),
                'stopPrice': r.get('stopPrice'),
            }
            out[_pending_order_key(row)] = row
        return out

    prev_map = _rows_map(prev_df)
    curr_map = _rows_map(curr_df)

    new_keys = sorted(set(curr_map.keys()) - set(prev_map.keys()))
    gone_keys = sorted(set(prev_map.keys()) - set(curr_map.keys()))

    for key in new_keys:
        r = curr_map[key]
        row = {
            'ts_utc': ts,
            'event_type': 'pending_seen',
            'symbol': r.get('symbol', ''),
            'order_id': _norm_order_id(r.get('orderId')),
            'type': r.get('type', ''),
            'side': r.get('side', ''),
            'position_side': r.get('positionSide', ''),
            'stop_price': _safe_float(r.get('stopPrice'), default=float('nan')),
            'source': 'query_pending_orders',
        }
        _append_csv_row(ORDER_LIFECYCLE_LOG_CSV, row, cols)

    for key in gone_keys:
        r = prev_map[key]
        row = {
            'ts_utc': ts,
            'event_type': 'pending_gone',
            'symbol': r.get('symbol', ''),
            'order_id': _norm_order_id(r.get('orderId')),
            'type': r.get('type', ''),
            'side': r.get('side', ''),
            'position_side': r.get('positionSide', ''),
            'stop_price': _safe_float(r.get('stopPrice'), default=float('nan')),
            'source': 'query_pending_orders',
        }
        _append_csv_row(ORDER_LIFECYCLE_LOG_CSV, row, cols)
        otype = str(r.get('type', '')).upper()
        symbol_u = str(r.get('symbol', '')).upper()
        pside_u = str(r.get('positionSide', '')).upper()
        side_u = str(r.get('side', '')).upper()
        order_id_norm = _norm_order_id(r.get('orderId'))

        # Patch 5C: TP LIMIT por etapas en modo partial_limit_tp (confirmacion inferida por reduccion de posicion).
        st_tp = get_tp_state(symbol_u, pside_u)
        tp_mode_state = str(st_tp.get("tp_mode", "")).lower()
        tp_fill_mode_state = str(st_tp.get("tp_fill_confirmation_mode", "inferred")).lower()
        tp_idx_state = _infer_tp_idx_from_state_order_id(st_tp, order_id_norm)
        is_stage_limit_candidate = (
            otype == "LIMIT"
            and tp_mode_state == "partial_limit_tp"
            and order_id_norm
            and tp_idx_state in (1, 2, 3)
        )
        if is_stage_limit_candidate:
            tp_idx = int(tp_idx_state)
            stage_name = _tp_stage_name(tp_idx)
            confirmed, confirm_reason, current_qty, reduction = _infer_tp_fill_from_position(symbol_u, pside_u, st_tp, tp_idx=tp_idx)
            confirm_source = "inferred_pending_gone_plus_position_reduction"
            if tp_fill_mode_state == "exchange_state":
                confirm_source = "exchange_state_unavailable_fallback_inferred"
            if confirmed:
                st = set_tp_filled(symbol_u, pside_u, tp_idx=tp_idx)
                emit_lifecycle_event(
                    f"{stage_name}_filled",
                    "INFO",
                    symbol=symbol_u,
                    position_side=pside_u,
                    side=side_u,
                    order_id=order_id_norm,
                    source=confirm_source,
                    confirmation_mode="inferred" if tp_fill_mode_state == "exchange_state" else tp_fill_mode_state,
                    reduction_qty=_safe_float_or_none(reduction),
                    remaining_qty=_safe_float_or_none(current_qty),
                    tp_stage=str(st.get("tp_stage", "")),
                )
                append_execution_ledger_event(
                    f"{stage_name}_filled",
                    data_quality="inferred",
                    source=f"pending_gone_limit_{stage_name}",
                    ts_utc=ts,
                    order_id=order_id_norm,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype,
                    submitted_price=_safe_float_or_none(st_tp.get(f"{stage_name}_price")),
                    fill_qty=_safe_float_or_none(st_tp.get(f"{stage_name}_qty")),
                    fill_time_utc=ts,
                    close_reason=stage_name,
                    partial_fill_status="inferred",
                    notes=f"{confirm_source}|{confirm_reason}",
                )
                if tp_idx == 3:
                    try:
                        _record_trade_closed(symbol_u, pside_u, "tp3")
                    except Exception:
                        pass
            else:
                _emit_tp_failed(
                    tp_idx=tp_idx,
                    symbol=symbol_u,
                    position_side=pside_u,
                    reason=f"{stage_name}_confirmation_failed",
                    detail=f"{confirm_source}|{confirm_reason}",
                    order_id=order_id_norm,
                    data_quality="inferred",
                    source=f"pending_gone_limit_{stage_name}",
                    tp_price=st_tp.get(f"{stage_name}_price"),
                    tp_qty=st_tp.get(f"{stage_name}_qty"),
                )
                if get_tp_legacy_fallback_on_error():
                    stage_qty = _safe_float_or_none(st_tp.get(f"{stage_name}_qty"))
                    if stage_qty is not None and current_qty is not None:
                        stage_qty = min(float(stage_qty), float(current_qty))
                    ok_fallback = _submit_tp_legacy_fallback(
                        tp_idx=tp_idx,
                        symbol=symbol_u,
                        position_side=pside_u,
                        market_ref=_last_traded_price(symbol_u),
                        tp_price=st_tp.get(f"{stage_name}_price"),
                        tp_qty=stage_qty,
                        tp_fill_mode=str(st_tp.get("tp_fill_confirmation_mode", "inferred")),
                        reason=f"{stage_name}_confirmation_failed:{confirm_reason}",
                    )
                    if not ok_fallback:
                        _emit_tp_failed(
                            tp_idx=tp_idx,
                            symbol=symbol_u,
                            position_side=pside_u,
                            reason="legacy_fallback_submit_failed",
                            detail=f"{stage_name}_confirmation_failed:{confirm_reason}",
                            order_id=order_id_norm,
                            data_quality="inferred",
                            source=f"pending_gone_limit_{stage_name}",
                            tp_price=st_tp.get(f"{stage_name}_price"),
                            tp_qty=stage_qty,
                        )
            continue

        if otype == 'TAKE_PROFIT_MARKET':
            tp_idx = _infer_tp_idx_from_state(symbol_u, pside_u, order_id_norm)
            confirmed, confirm_reason, current_qty, reduction = _infer_tp_fill_from_position(
                symbol_u,
                pside_u,
                st_tp,
                tp_idx=tp_idx,
            )
            if confirmed:
                emit_lifecycle_event(
                    "take_profit_hit",
                    "INFO",
                    symbol=symbol_u,
                    position_side=pside_u,
                    side=side_u,
                    order_id=order_id_norm,
                    source="inferred_pending_gone_plus_position_reduction",
                )
                append_execution_ledger_event(
                    "take_profit_hit",
                    data_quality="inferred",
                    source="pending_gone_plus_position_reduction",
                    ts_utc=ts,
                    order_id=order_id_norm,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype,
                    stop_price=_safe_float_or_none(r.get('stopPrice')),
                    fill_time_utc=ts,
                    close_reason="take_profit",
                    partial_fill_status="inferred",
                    notes=f"tp{tp_idx}|{confirm_reason}",
                )
                st = set_tp_filled(symbol_u, pside_u, tp_idx=tp_idx)
                emit_lifecycle_event(
                    f"tp{tp_idx}_filled",
                    "INFO",
                    symbol=symbol_u,
                    position_side=pside_u,
                    order_id=order_id_norm,
                    source="inferred_pending_gone_plus_position_reduction",
                    tp_stage=str(st.get("tp_stage", "")),
                    reduction_qty=_safe_float_or_none(reduction),
                    remaining_qty=_safe_float_or_none(current_qty),
                )
                append_execution_ledger_event(
                    f"tp{tp_idx}_filled",
                    data_quality="inferred",
                    source="pending_gone_plus_position_reduction",
                    order_id=order_id_norm,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype,
                    fill_time_utc=ts,
                    close_reason=f"tp{tp_idx}",
                    partial_fill_status="inferred",
                    notes=confirm_reason,
                )
                if tp_idx == 3:
                    try:
                        _record_trade_closed(symbol_u, pside_u, "tp3")
                    except Exception:
                        pass
            else:
                emit_lifecycle_event(
                    "execution_quality_warning",
                    "WARN",
                    symbol=symbol_u,
                    position_side=pside_u,
                    side=side_u,
                    order_id=order_id_norm,
                    reason="tp_pending_gone_not_confirmed",
                    detail=str(confirm_reason)[:220],
                    source="pending_gone_confirmation_guard",
                )
                append_execution_ledger_event(
                    "tp_confirmation_failed",
                    data_quality="inferred",
                    source="pending_gone_confirmation_guard",
                    ts_utc=ts,
                    order_id=order_id_norm,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype,
                    stop_price=_safe_float_or_none(r.get('stopPrice')),
                    fill_time_utc=ts,
                    close_reason=f"tp{tp_idx}_not_confirmed",
                    partial_fill_status="unknown",
                    notes=str(confirm_reason)[:220],
                )

def _normalize_orders_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas coherentes: ['symbol','orderId','type','stopPrice','time'] cuando existan.
    Mapea 'orderType'→'type', 'symbol' en mayúsculas y 'stopPrice' numérico.
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
    df = df.copy()
    if 'type' not in df.columns and 'orderType' in df.columns:
        df = df.rename(columns={'orderType': 'type'})
    if 'type' not in df.columns:
        df['type'] = ''
    if 'orderId' not in df.columns:
        df['orderId'] = ''
    if 'side' not in df.columns:
        df['side'] = ''
    if 'positionSide' not in df.columns:
        df['positionSide'] = ''
    if 'price' not in df.columns:
        df['price'] = ''
    df['orderId'] = df['orderId'].apply(_norm_order_id)
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].astype(str).str.upper()
    df['side'] = df['side'].astype(str).str.upper().str.strip()
    df['positionSide'] = df['positionSide'].astype(str).str.upper().str.strip()
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'stopPrice' in df.columns:
        df['stopPrice'] = pd.to_numeric(df['stopPrice'], errors='coerce')
    return df


def _load_orders_register_df() -> pd.DataFrame:
    """Carga `order_id_register.csv` normalizado con fallback a DataFrame vacío."""
    try:
        df = pd.read_csv('./archivos/order_id_register.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['symbol', 'orderId', 'type', 'stopPrice', 'time'])
    except Exception:
        df = pd.DataFrame(columns=['symbol', 'orderId', 'type', 'stopPrice', 'time'])
    return _normalize_orders_df(df)

def _cooldown_minutes_for_symbol(params_by_symbol: dict, symbol: str, default: int = 10) -> int:
    fixed_cd = get_cooldown_minutes_override()
    if fixed_cd is not None:
        return int(fixed_cd)
    try:
        p = params_by_symbol.get(str(symbol).upper(), {}) if isinstance(params_by_symbol, dict) else {}
        cd = p.get('cooldown', p.get('cooldown_min', default))
        return int(cd)
    except Exception:
        return default

def _write_cooldown(symbol: str, ts, minutes: int = 10) -> None:
    try:
        if ts is None:
            ts = datetime.utcnow()
        expiry = ts + timedelta(minutes=int(minutes))
        df_exist = pd.read_csv(COOLDOWN_CSV, parse_dates=['expires_at']) if os.path.exists(COOLDOWN_CSV) else pd.DataFrame(columns=['symbol','expires_at'])
        df_exist['symbol'] = df_exist['symbol'].astype(str).str.upper()
        symbol = str(symbol).upper()
        if symbol in df_exist['symbol'].values:
            df_exist.loc[df_exist['symbol'] == symbol, 'expires_at'] = expiry
        else:
            df_exist = pd.concat([df_exist, pd.DataFrame({'symbol':[symbol], 'expires_at':[expiry]})], ignore_index=True)
        df_exist.to_csv(COOLDOWN_CSV, index=False)
    except Exception as e:
        print(f"Error registrando cooldown para {symbol}: {e}")


def _load_active_cooldowns(now=None) -> dict:
    now = now or datetime.utcnow()
    try:
        df = pd.read_csv(COOLDOWN_CSV, parse_dates=['expires_at'])
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Error leyendo cooldowns: {e}")
        return {}
    df['symbol'] = df['symbol'].astype(str).str.upper()
    df = df[df['expires_at'] > now]
    try:
        df.to_csv(COOLDOWN_CSV, index=False)
    except Exception:
        pass
    return {row['symbol']: row['expires_at'].to_pydatetime() for _, row in df.iterrows()}


def _append_sl_watch(symbol: str, stop_price: float, position_side: str, order_id) -> None:
    try:
        df = pd.read_csv(SL_WATCH_CSV) if os.path.exists(SL_WATCH_CSV) else pd.DataFrame(columns=['symbol','stop_price','position_side','orderId','ts'])
        df['symbol'] = df['symbol'].astype(str).str.upper()
        symbol = str(symbol).upper()
        row = {
            'symbol': symbol,
            'stop_price': float(stop_price) if stop_price is not None else None,
            'position_side': str(position_side).upper(),
            'orderId': str(order_id) if order_id not in (None, '', float('nan')) else '',
            'ts': datetime.utcnow().isoformat()
        }
        df = df[~((df['symbol'] == symbol) & (df['position_side'] == row['position_side']))]
        df.loc[len(df)] = row
        df.to_csv(SL_WATCH_CSV, index=False)
    except Exception as e:
        print(f"Error registrando SL watch para {symbol}: {e}")


def _append_entry_watch(
    symbol: str,
    position_side: str,
    qty: float,
    request_id: str = "",
    order_id: str = "",
    side: str = "",
    intended_entry_price=None,
    submitted_price=None,
    submit_time_utc: str = "",
    trade_capital: float = 0.0,
    peso_pct: float = 0.0,
) -> None:
    cols = [
        'symbol',
        'position_side',
        'side',
        'qty',
        'request_id',
        'order_id',
        'intended_entry_price',
        'submitted_price',
        'submit_time_utc',
        'trade_capital',
        'peso_pct',
        'ts_utc',
    ]
    try:
        df = pd.read_csv(ENTRY_WATCH_CSV) if os.path.exists(ENTRY_WATCH_CSV) else pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df.columns:
                df[c] = ''
        symbol_u = str(symbol).upper().strip()
        pside_u = str(position_side).upper().strip()
        df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
        df['position_side'] = df['position_side'].astype(str).str.upper().str.strip()
        df = df[~((df['symbol'] == symbol_u) & (df['position_side'] == pside_u))]
        row = {
            'symbol': symbol_u,
            'position_side': pside_u,
            'side': str(side or '').upper().strip(),
            'qty': _safe_float(qty, 0.0),
            'request_id': str(request_id or '').strip(),
            'order_id': _norm_order_id(order_id),
            'intended_entry_price': _safe_float_or_none(intended_entry_price),
            'submitted_price': _safe_float_or_none(submitted_price),
            'submit_time_utc': str(submit_time_utc or '').strip(),
            'trade_capital': _safe_float(trade_capital, 0.0),
            'peso_pct': _safe_float(peso_pct, 0.0),
            'ts_utc': _utc_now_iso(),
        }
        df = pd.concat([df, pd.DataFrame([row], columns=cols)], ignore_index=True)
        df.to_csv(ENTRY_WATCH_CSV, index=False)
    except Exception as e:
        print(f"Error registrando entry watch para {symbol}: {e}")


def _consume_entry_watch(symbol: str, position_side: str):
    cols = [
        'symbol',
        'position_side',
        'side',
        'qty',
        'request_id',
        'order_id',
        'intended_entry_price',
        'submitted_price',
        'submit_time_utc',
        'ts_utc',
    ]
    if not os.path.exists(ENTRY_WATCH_CSV):
        return None
    try:
        df = pd.read_csv(ENTRY_WATCH_CSV)
    except Exception:
        return None
    if df.empty:
        return None
    for c in cols:
        if c not in df.columns:
            df[c] = ''
    df['symbol'] = df['symbol'].astype(str).str.upper().str.strip()
    df['position_side'] = df['position_side'].astype(str).str.upper().str.strip()
    symbol_u = str(symbol).upper().strip()
    pside_u = str(position_side).upper().strip()
    mask = (df['symbol'] == symbol_u) & (df['position_side'] == pside_u)
    if not mask.any():
        return None
    hit = df[mask].tail(1).iloc[0].to_dict()
    remaining = df[~mask].copy()
    try:
        if remaining.empty:
            os.remove(ENTRY_WATCH_CSV)
        else:
            remaining.to_csv(ENTRY_WATCH_CSV, index=False)
    except Exception:
        pass
    return hit
    
 # Paso mínimo global de lote (0.001).  Si en el futuro necesitas pasos específicos,
# vuelve a añadir un diccionario STEP_SIZES y usa .get().

STEP_SIZE_DEFAULT = 0.001
# Overrides por símbolo para step/tick conocidos (completar según reglas del exchange).
SYMBOL_TRADING_RULES = {
    'BNB-USDT': {'qty_step': 0.01, 'price_tick': 0.1},
    'DOT-USDT': {'qty_step': 0.1, 'price_tick': 0.001},
    'CFX-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
    'HBAR-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
    # TRX y DOGE requieren ticks finos; con 0.01 el TP quedaba por debajo del precio de entrada
    'TRX-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
    'DOGE-USDT': {'qty_step': 1.0, 'price_tick': 0.0001},
}
# Splits legacy por defecto para TP escalonado (40%, 40%, 20%)
TP_SPLITS = (0.40, 0.40, 0.20)


def _runtime_tp_splits():
    """Obtiene splits TP desde runtime config con fallback legacy."""
    try:
        raw = tuple(float(x) for x in get_tp_partial_distribution())
    except Exception:
        raw = TP_SPLITS
    if len(raw) < 3:
        return TP_SPLITS
    vals = [max(float(v), 0.0) for v in raw[:3]]
    total = sum(vals)
    if total <= 0:
        return TP_SPLITS
    return tuple(float(v / total) for v in vals)


def _tp_stage_from_live_count(live_count: int) -> str:
    n = int(live_count)
    if n >= 3:
        return "tp3_live"
    if n >= 2:
        return "tp2_live"
    if n >= 1:
        return "tp1_live"
    return "none"


def _infer_tp_idx_from_state(symbol: str, position_side: str, order_id: str) -> int:
    """Best-effort: identifica TP1/TP2/TP3 por order_id o etapa actual."""
    st = get_tp_state(symbol, position_side)
    oid = _norm_order_id(order_id)
    if oid:
        if oid and oid == _norm_order_id(st.get("tp1_order_id")):
            return 1
        if oid and oid == _norm_order_id(st.get("tp2_order_id")):
            return 2
        if oid and oid == _norm_order_id(st.get("tp3_order_id")):
            return 3
    stage = str(st.get("tp_stage", "none")).lower()
    if stage in ("tp3_live", "tp2_filled"):
        return 3
    if stage in ("tp2_live", "tp1_filled"):
        return 2
    return 1

def _round_step(qty: float, step: float = STEP_SIZE_DEFAULT) -> float:
    """Redondea cantidad al múltiplo permitido por el contrato."""
    if step <= 0:
        return float(qty)
    raw = math.floor(float(qty) / step) * step
    decs = max(0, -int(round(math.log10(step))))
    return round(raw, decs)


def _split_position_qtys(total_qty: float, splits, step: float = STEP_SIZE_DEFAULT):
    """Divide la posición entre splits, garantizando que la suma use todo el tamaño en múltiplos de step."""
    if total_qty is None:
        return [0.0 for _ in splits]
    try:
        qty_dec = Decimal(str(abs(float(total_qty))))
    except Exception:
        return [0.0 for _ in splits]
    if qty_dec <= 0:
        return [0.0 for _ in splits]

    step = STEP_SIZE_DEFAULT if step is None else step
    if step <= 0:
        step = STEP_SIZE_DEFAULT
    step_dec = Decimal(str(step))

    # Ajustar cantidad total al múltiplo válido más cercano hacia abajo
    qty_dec = (qty_dec / step_dec).to_integral_value(rounding=ROUND_DOWN) * step_dec
    units_total = int((qty_dec / step_dec))
    if units_total <= 0:
        return [0.0 for _ in splits]

    result = []
    remaining_units = units_total
    splits = tuple(splits) if splits else (1.0,)

    for idx, split in enumerate(splits):
        if idx == len(splits) - 1:
            units = remaining_units
        else:
            try:
                fraction = Decimal(str(split))
            except Exception:
                fraction = Decimal('0')
            if fraction <= 0:
                units = 0
            else:
                units = int((fraction * Decimal(units_total)).to_integral_value(rounding=ROUND_DOWN))
            if units > remaining_units:
                units = remaining_units
        result.append(float(step_dec * units))
        remaining_units -= units
    # Si por redondeos quedó remanente, acumularlo al último TP
    if remaining_units > 0:
        result[-1] += float(step_dec * remaining_units)
    return result


def _step_size_for(symbol: str) -> float:
    return SYMBOL_TRADING_RULES.get(str(symbol).upper(), {}).get('qty_step', STEP_SIZE_DEFAULT)


def _tick_size_for(symbol: str) -> float:
    return SYMBOL_TRADING_RULES.get(str(symbol).upper(), {}).get('price_tick', 0.01)


def _round_to_tick(value: float, tick: float, rounding=ROUND_DOWN) -> float:
    if tick is None or tick <= 0:
        return float(value)
    try:
        tick_dec = Decimal(str(tick))
        val_dec = Decimal(str(value))
        rounded = (val_dec / tick_dec).to_integral_value(rounding=rounding) * tick_dec
        return float(rounded)
    except Exception:
        return float(value)


def _trigger_rounding(position_side: str, order_type: str):
    side = str(position_side).upper()
    otype = str(order_type).upper()
    if otype == "TAKE_PROFIT_MARKET":
        return ROUND_UP if side == "LONG" else ROUND_DOWN
    if otype == "STOP_MARKET":
        return ROUND_DOWN if side == "LONG" else ROUND_UP
    return ROUND_DOWN


def _round_trigger_price(raw_price: float, symbol: str, position_side: str, order_type: str) -> float:
    tick = _tick_size_for(symbol)
    return _round_to_tick(float(raw_price), tick, rounding=_trigger_rounding(position_side, order_type))


def _last_traded_price(symbol: str):
    try:
        raw = pkg.bingx.last_price_trading_par(symbol)
        data = json.loads(raw) if isinstance(raw, str) else raw
        payload = data.get('data', {}) if isinstance(data, dict) else {}
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        for key in ('price', 'close', 'markPrice', 'lastPrice'):
            if key in payload:
                return float(payload[key])
    except Exception:
        return None
    return None


def _sanitize_trigger_price(raw_price: float, symbol: str, position_side: str, order_type: str, ref_price) -> float:
    side = str(position_side).upper()
    otype = str(order_type).upper()
    tick = _tick_size_for(symbol)
    min_px = tick if tick and tick > 0 else 1e-8

    px = _round_trigger_price(float(raw_price), symbol, side, otype)
    try:
        ref = float(ref_price)
    except Exception:
        ref = None

    if ref is None or ref <= 0:
        return float(px if px > 0 else min_px)

    if otype == "TAKE_PROFIT_MARKET":
        if side == "LONG" and px <= ref:
            px = _round_trigger_price(ref + tick, symbol, side, otype)
        elif side == "SHORT" and px >= ref:
            px = _round_trigger_price(max(ref - tick, ref * 0.999), symbol, side, otype)
    elif otype == "STOP_MARKET":
        if side == "LONG" and px >= ref:
            px = _round_trigger_price(max(ref - tick, ref * 0.999), symbol, side, otype)
        elif side == "SHORT" and px <= ref:
            px = _round_trigger_price(ref + tick, symbol, side, otype)

    if px <= 0:
        px = min_px
    return float(px)


def _bootstrap_symbols_for_protection() -> list:
    symbols = set()
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            for row in _prod:
                if isinstance(row, dict):
                    sym = str(row.get('symbol', '')).upper().strip()
                    if sym:
                        symbols.add(sym)
    except Exception:
        pass

    return sorted(symbols)


def _bootstrap_position_queue(df_posiciones: pd.DataFrame, df_ordenes: pd.DataFrame) -> pd.DataFrame:
    if df_posiciones is None:
        df_posiciones = pd.DataFrame(columns=['symbol','tipo','counter'])
    if not df_posiciones.empty:
        return df_posiciones

    symbols = _bootstrap_symbols_for_protection()
    if not symbols:
        return df_posiciones

    new_rows = []
    df_ordenes = _normalize_orders_df(df_ordenes)
    for symbol in symbols:
        try:
            symbol_result, position_side, _price, position_amt, _upnl = total_positions(symbol)
        except Exception:
            continue
        if symbol_result is None or position_side not in ('LONG', 'SHORT'):
            continue
        try:
            qty = abs(float(position_amt))
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue

        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
        sl_exists = not symbol_orders[symbol_orders['type'] == 'STOP_MARKET'].empty
        tp_exists = not symbol_orders[symbol_orders['type'] == 'TAKE_PROFIT_MARKET'].empty
        if sl_exists and tp_exists:
            continue

        new_rows.append({'symbol': symbol, 'tipo': position_side, 'counter': 0})

    if not new_rows:
        return df_posiciones

    bootstrap_df = pd.DataFrame(new_rows, columns=['symbol', 'tipo', 'counter'])
    print(f"Arranque en frio: se agregaron {len(bootstrap_df)} posiciones abiertas a la cola de proteccion.")
    return pd.concat([df_posiciones, bootstrap_df], ignore_index=True)

# --- Retry helper for robust order posting ---
def _post_with_retry(
    symbol,
    qty,
    price,
    stop,
    position_side,
    order_type,
    side,
    delays=(0.5, 1.0, 2.0),
    order_kwargs=None,
    intended_entry_price=None,
    emit_entry_lifecycle: bool = False,
    return_details: bool = False,
):
    """Intenta colocar una orden con reintentos escalonados.

    Por compatibilidad:
    - return_details=False -> retorna bool.
    - return_details=True  -> retorna (ok: bool, details: dict).
    """
    last_err = None
    last_code = ''
    last_msg = ''
    last_order_id = ''
    request_id = uuid.uuid4().hex
    order_kwargs = dict(order_kwargs or {})

    symbol_u = str(symbol).upper()
    side_u = str(side).upper()
    pside_u = str(position_side).upper()
    otype_u = str(order_type).upper()

    details = {
        'request_id': request_id,
        'order_id': '',
        'symbol': symbol_u,
        'side': side_u,
        'position_side': pside_u,
        'order_type': otype_u,
        'submit_qty': _safe_float(qty, 0.0),
        'submitted_price': _safe_float_or_none(price),
        'stop_price': _safe_float_or_none(stop),
        'intended_entry_price': _safe_float_or_none(intended_entry_price),
        'submit_time_utc': '',
        'attempt': 0,
        'accepted': False,
        'code': '',
        'msg': '',
    }

    for idx, d in enumerate(delays, start=1):
        try:
            resp = pkg.bingx.post_order(
                symbol,
                qty,
                price,
                stop,
                position_side,
                order_type,
                side,
                **order_kwargs,
            )
            code = None
            msg = None
            order_id = None
            try:
                parsed = json.loads(resp) if isinstance(resp, str) else resp
                if isinstance(parsed, dict):
                    code = parsed.get('code')
                    msg = parsed.get('msg') or parsed.get('message') or ''
                    order_id = _extract_order_id_from_payload(parsed.get('data', parsed))
                else:
                    msg = f"respuesta_no_dict:{type(parsed).__name__}"
            except Exception as pe:
                msg = f"respuesta_no_json:{pe}"

            ok, err_msg = _validate_order_response(resp)
            _log_order_submit_attempt(
                request_id=request_id,
                symbol=symbol,
                qty=qty,
                price=price,
                stop_price=stop,
                position_side=position_side,
                order_type=order_type,
                side=side,
                attempt=idx,
                delay_s=d,
                accepted=ok,
                code=code,
                msg=(err_msg or msg or ''),
                order_id=order_id,
                raw_response=resp,
            )
            if ok:
                submit_ts_utc = _utc_now_iso()
                order_id_norm = _norm_order_id(order_id)
                details.update(
                    {
                        'order_id': order_id_norm,
                        'submit_time_utc': submit_ts_utc,
                        'attempt': idx,
                        'accepted': True,
                        'code': '' if code is None else str(code),
                        'msg': (err_msg or msg or ''),
                    }
                )
                append_execution_ledger_event(
                    "order_submitted",
                    data_quality="actual",
                    source="post_with_retry_accept",
                    request_id=request_id,
                    order_id=order_id_norm,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype_u,
                    intended_entry_price=_safe_float_or_none(intended_entry_price),
                    submitted_price=_safe_float_or_none(price),
                    stop_price=_safe_float_or_none(stop),
                    submit_qty=_safe_float_or_none(qty),
                    submit_time_utc=submit_ts_utc,
                    raw_code=code,
                    raw_msg=(err_msg or msg or ''),
                    notes=f"attempt={idx}",
                )
                if otype_u == "MARKET" or emit_entry_lifecycle:
                    emit_lifecycle_event(
                        "entry_order_submitted",
                        "INFO",
                        symbol=symbol_u,
                        position_side=pside_u,
                        side=side_u,
                        qty=_safe_float(qty, 0.0),
                        attempt=idx,
                        request_id=request_id,
                        order_id=order_id_norm,
                        source="api_submit_accepted",
                    )
                if return_details:
                    return True, details
                return True
            # Fallback seguro: en Hedge mode, algunas cuentas rechazan reduceOnly en MARKET.
            # Reintentamos una vez sin reduceOnly para evitar fallo operativo en cierres de emergencia.
            fallback_attempted = False
            if (
                otype_u == "MARKET"
                and "reduceOnly" in order_kwargs
                and _is_reduce_only_hedge_rejection(code, (err_msg or msg or ""))
            ):
                fallback_attempted = True
                order_kwargs.pop("reduceOnly", None)
                warn_detail = "reduceOnly_rejected_in_hedge_mode_retry_without_reduceOnly"
                emit_lifecycle_event(
                    "execution_quality_warning",
                    "WARN",
                    symbol=symbol_u,
                    position_side=pside_u,
                    side=side_u,
                    order_type=otype_u,
                    reason="reduce_only_rejected_hedge_mode",
                    attempt=idx,
                    source="post_with_retry_fallback",
                )
                append_execution_ledger_event(
                    "order_submit_retry_without_reduce_only",
                    data_quality="actual",
                    source="post_with_retry_fallback",
                    request_id=request_id,
                    symbol=symbol_u,
                    side=side_u,
                    position_side=pside_u,
                    order_type=otype_u,
                    submit_qty=_safe_float_or_none(qty),
                    raw_code=code,
                    raw_msg=(err_msg or msg or ""),
                    notes=warn_detail,
                )
                try:
                    resp2 = pkg.bingx.post_order(
                        symbol,
                        qty,
                        price,
                        stop,
                        position_side,
                        order_type,
                        side,
                        **order_kwargs,
                    )
                    code2 = None
                    msg2 = None
                    order_id2 = None
                    try:
                        parsed2 = json.loads(resp2) if isinstance(resp2, str) else resp2
                        if isinstance(parsed2, dict):
                            code2 = parsed2.get('code')
                            msg2 = parsed2.get('msg') or parsed2.get('message') or ''
                            order_id2 = _extract_order_id_from_payload(parsed2.get('data', parsed2))
                        else:
                            msg2 = f"respuesta_no_dict:{type(parsed2).__name__}"
                    except Exception as pe2:
                        msg2 = f"respuesta_no_json:{pe2}"

                    ok2, err_msg2 = _validate_order_response(resp2)
                    _log_order_submit_attempt(
                        request_id=request_id,
                        symbol=symbol,
                        qty=qty,
                        price=price,
                        stop_price=stop,
                        position_side=position_side,
                        order_type=order_type,
                        side=side,
                        attempt=idx,
                        delay_s=0.0,
                        accepted=ok2,
                        code=code2,
                        msg=(err_msg2 or msg2 or ''),
                        order_id=order_id2,
                        raw_response=resp2,
                    )
                    if ok2:
                        submit_ts_utc = _utc_now_iso()
                        order_id_norm = _norm_order_id(order_id2)
                        details.update(
                            {
                                'order_id': order_id_norm,
                                'submit_time_utc': submit_ts_utc,
                                'attempt': idx,
                                'accepted': True,
                                'code': '' if code2 is None else str(code2),
                                'msg': (err_msg2 or msg2 or ''),
                            }
                        )
                        append_execution_ledger_event(
                            "order_submitted",
                            data_quality="actual",
                            source="post_with_retry_accept_after_reduce_only_fallback",
                            request_id=request_id,
                            order_id=order_id_norm,
                            symbol=symbol_u,
                            side=side_u,
                            position_side=pside_u,
                            order_type=otype_u,
                            intended_entry_price=_safe_float_or_none(intended_entry_price),
                            submitted_price=_safe_float_or_none(price),
                            stop_price=_safe_float_or_none(stop),
                            submit_qty=_safe_float_or_none(qty),
                            submit_time_utc=submit_ts_utc,
                            raw_code=code2,
                            raw_msg=(err_msg2 or msg2 or ''),
                            notes=f"attempt={idx},retry_without_reduce_only=1",
                        )
                        if otype_u == "MARKET" or emit_entry_lifecycle:
                            emit_lifecycle_event(
                                "entry_order_submitted",
                                "INFO",
                                symbol=symbol_u,
                                position_side=pside_u,
                                side=side_u,
                                qty=_safe_float(qty, 0.0),
                                attempt=idx,
                                request_id=request_id,
                                order_id=order_id_norm,
                                source="api_submit_accepted_after_reduce_only_fallback",
                            )
                        if return_details:
                            return True, details
                        return True
                    last_err = err_msg2
                    last_code = '' if code2 is None else str(code2)
                    last_msg = (err_msg2 or msg2 or '')
                    last_order_id = _norm_order_id(order_id2)
                except Exception as e2:
                    last_err = e2
                    last_code = 'EXCEPTION'
                    last_msg = str(e2)
                    _log_order_submit_attempt(
                        request_id=request_id,
                        symbol=symbol,
                        qty=qty,
                        price=price,
                        stop_price=stop,
                        position_side=position_side,
                        order_type=order_type,
                        side=side,
                        attempt=idx,
                        delay_s=0.0,
                        accepted=False,
                        code='EXCEPTION',
                        msg=str(e2),
                        order_id='',
                        raw_response='',
                    )
            if not fallback_attempted:
                last_err = err_msg
                last_code = '' if code is None else str(code)
                last_msg = (err_msg or msg or '')
                last_order_id = _norm_order_id(order_id)
            reject_detail = (str(last_err) if fallback_attempted else str(err_msg))
            print(f"post_order({order_type}) rechazado para {symbol}: {reject_detail}")
        except Exception as e:
            last_err = e
            last_code = 'EXCEPTION'
            last_msg = str(e)
            _log_order_submit_attempt(
                request_id=request_id,
                symbol=symbol,
                qty=qty,
                price=price,
                stop_price=stop,
                position_side=position_side,
                order_type=order_type,
                side=side,
                attempt=idx,
                delay_s=d,
                accepted=False,
                code='EXCEPTION',
                msg=str(e),
                order_id='',
                raw_response='',
            )
        time.sleep(d)

    print(f"Fallo post_order({order_type}) para {symbol}: {last_err}")
    details.update(
        {
            'order_id': last_order_id,
            'attempt': len(delays),
            'accepted': False,
            'code': last_code,
            'msg': str(last_err)[:240] if last_err is not None else '',
        }
    )
    append_execution_ledger_event(
        "order_submit_failed",
        data_quality="actual",
        source="post_with_retry_exhausted",
        request_id=request_id,
        order_id=last_order_id,
        symbol=symbol_u,
        side=side_u,
        position_side=pside_u,
        order_type=otype_u,
        intended_entry_price=_safe_float_or_none(intended_entry_price),
        submitted_price=_safe_float_or_none(price),
        stop_price=_safe_float_or_none(stop),
        submit_qty=_safe_float_or_none(qty),
        raw_code=last_code,
        raw_msg=str(last_err)[:240] if last_err is not None else '',
        cancel_reason="submit_rejected_or_exception",
    )
    if otype_u == "MARKET" or emit_entry_lifecycle:
        emit_lifecycle_event(
            "entry_order_canceled_or_expired",
            "WARN",
            symbol=symbol_u,
            position_side=pside_u,
            side=side_u,
            qty=_safe_float(qty, 0.0),
            reason="entry_submit_rejected_or_exception",
            detail=str(last_err)[:220] if last_err is not None else "unknown",
            source="post_with_retry_exhausted",
        )
        append_execution_ledger_event(
            "entry_order_canceled_or_expired",
            data_quality="actual",
            source="post_with_retry_exhausted",
            request_id=request_id,
            order_id=last_order_id,
            symbol=symbol_u,
            side=side_u,
            position_side=pside_u,
            order_type=otype_u,
            intended_entry_price=_safe_float_or_none(intended_entry_price),
            submitted_price=_safe_float_or_none(price),
            submit_qty=_safe_float_or_none(qty),
            cancel_reason="entry_submit_rejected_or_exception",
            raw_code=last_code,
            raw_msg=str(last_err)[:240] if last_err is not None else '',
        )
    if return_details:
        return False, details
    return False


def _validate_order_response(resp_text):
    try:
        data = json.loads(resp_text) if isinstance(resp_text, str) else resp_text
    except Exception as e:
        return False, f"Respuesta no JSON ({e}): {resp_text}"

    if not isinstance(data, dict):
        return False, f"Formato inesperado: {data}"

    code = data.get('code')
    if code in (0, '0'):
        return True, None

    msg = data.get('msg') or data.get('message') or 'sin detalle'
    return False, f"code={code}, msg={msg}"


def _is_reduce_only_hedge_rejection(code, err_msg: str) -> bool:
    """Detecta rechazo típico de BingX en Hedge mode cuando se envía reduceOnly."""
    c = str(code or "").strip()
    msg = str(err_msg or "").lower()
    if c != "109400":
        return False
    return ("reduceonly" in msg) or ("reduce only" in msg) or ("hedge mode" in msg)

import pkg.price_bingx_5m

#Obtener los valores de SL TP


def get_last_take_profit_stop_loss(symbol):
    # Leer el archivo CSV con los indicadores
    df = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    df['symbol'] = df['symbol'].str.upper()

    filtered_df = df[df['symbol'] == symbol.upper()]
    if filtered_df.empty:
        return None, None

    # Preferimos columnas LONG; si no existen, usamos SHORT.
    if {'Take_Profit_Long', 'Stop_Loss_Long'}.issubset(filtered_df.columns):
        return (
            filtered_df['Take_Profit_Long'].iloc[-1],
            filtered_df['Stop_Loss_Long'].iloc[-1],
        )
    elif {'Take_Profit_Short', 'Stop_Loss_Short'}.issubset(filtered_df.columns):
        return (
            filtered_df['Take_Profit_Short'].iloc[-1],
            filtered_df['Stop_Loss_Short'].iloc[-1],
        )

    # Si no existen columnas estándar, retorna None
    return None, None


# Extraer TP ladder y SL de indicadores para un símbolo y lado
def extract_tp_sl_from_latest(latest_values: pd.DataFrame, symbol: str, side: str):
    """
    Devuelve (tp_levels:list, sl_level:float) para el símbolo y lado dados usando columnas de indicadores.
    Si no hay columnas escalonadas, intenta usar el TP clásico como único nivel.
    side ∈ {"LONG","SHORT"}
    """
    side = str(side).upper()
    sym = str(symbol).upper()
    row = latest_values[latest_values['symbol'] == sym]
    if row.empty:
        return [], None
    r = row.iloc[0]
    tps = []
    sl = None
    if side == 'LONG':
        # Preferir escalonados
        if all(col in row.columns for col in ['TP1_L','TP2_L','TP3_L']):
            tps = [float(r['TP1_L']), float(r['TP2_L']), float(r['TP3_L'])]
        elif 'Take_Profit_Long' in row.columns:
            tps = [float(r['Take_Profit_Long'])]
        # SL
        if 'Stop_Loss_Long' in row.columns:
            sl = float(r['Stop_Loss_Long'])
    else:  # SHORT
        if all(col in row.columns for col in ['TP1_S','TP2_S','TP3_S']):
            tps = [float(r['TP1_S']), float(r['TP2_S']), float(r['TP3_S'])]
        elif 'Take_Profit_Short' in row.columns:
            tps = [float(r['Take_Profit_Short'])]
        if 'Stop_Loss_Short' in row.columns:
            sl = float(r['Stop_Loss_Short'])
    # Limpieza
    tps = [float(x) for x in tps if pd.notna(x)]
    sl = float(sl) if sl is not None and pd.notna(sl) else None
    return tps, sl


def _build_fill_alert(
    symbol: str,
    position_side: str,
    fill_price: float,
    tps: list,
    sl_level,
    trade_capital: float = 0.0,
    peso_pct: float = 0.0,
) -> str:
    """Mensaje unificado de operación confirmada para Telegram."""
    side_emoji = '🟢' if position_side == 'LONG' else '🔴'
    side_label = position_side.upper()
    # Nombre limpio del par (sin -USDT)
    pair_name = symbol.replace('-USDT', '')

    sign = 1.0 if position_side == 'LONG' else -1.0

    def _pct(px):
        try:
            return f"{sign * (px / fill_price - 1.0) * 100:+.2f}%"
        except Exception:
            return ""

    # Entrada
    lines = [
        f"{'━' * 14}",
        f"{side_emoji} *{pair_name}* {side_label}",
        f"{'━' * 14}",
        "",
        f"Entrada  `{fill_price:.4f}`",
    ]

    # TPs
    for i, tp_px in enumerate(tps[:3], start=1):
        tp_f = float(tp_px)
        lines.append(f"TP{i}       `{tp_f:.4f}`  {_pct(tp_f)}")

    # SL
    if sl_level is not None:
        sl_f = float(sl_level)
        lines.append(f"SL        `{sl_f:.4f}`  {_pct(sl_f)}")

    # R:R (riesgo vs TP1)
    if tps and sl_level is not None:
        try:
            risk = abs(fill_price - float(sl_level))
            reward = abs(float(tps[0]) - fill_price)
            if risk > 0:
                rr = reward / risk
                lines.append(f"R:R      `1:{rr:.1f}`")
        except Exception:
            pass

    lines.append("")

    # Capital
    if trade_capital > 0:
        lines.append(f"💰 `${trade_capital:.2f}` ({peso_pct:.0f}%)")

    # Splits
    try:
        s1, s2, s3 = _runtime_tp_splits()
        if len(tps) >= 3:
            lines.append(f"📊 `{s1*100:.0f}/{s2*100:.0f}/{s3*100:.0f}`")
    except Exception:
        pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Mensaje: Trade cerrado (se llama cuando se detecta SL hit o TP3 filled)
# ---------------------------------------------------------------------------
_TRADE_CLOSED_CSV = './archivos/trade_closed_log.csv'


def _record_trade_closed(symbol: str, position_side: str, close_reason: str):
    """Registra el cierre y envía alerta con PnL y duración."""
    try:
        pair_name = symbol.replace('-USDT', '')
        symbol_u = str(symbol).upper().strip()
        pside_u = str(position_side).upper().strip()

        # Buscar entrada en execution_ledger
        ledger_path = './archivos/execution_ledger.csv'
        entry_price = None
        entry_time = None
        if os.path.exists(ledger_path):
            df_led = pd.read_csv(ledger_path, low_memory=False)
            mask = (
                (df_led['symbol'].astype(str).str.upper() == symbol_u)
                & (df_led['position_side'].astype(str).str.upper() == pside_u)
                & (df_led['event_type'].astype(str).str.strip() == 'entry_order_filled')
            )
            entries = df_led[mask].copy()
            if not entries.empty:
                last_entry = entries.iloc[-1]
                entry_price = pd.to_numeric(last_entry.get('actual_fill_price'), errors='coerce')
                if pd.isna(entry_price):
                    entry_price = pd.to_numeric(last_entry.get('intended_entry_price'), errors='coerce')
                entry_time = pd.to_datetime(last_entry.get('ts_utc'), errors='coerce', utc=True)

        # PnL desde PnL.csv (últimas 24h para este símbolo)
        pnl_val = 0.0
        pnl_csv = './archivos/PnL.csv'
        if os.path.exists(pnl_csv):
            df_pnl = pd.read_csv(pnl_csv, low_memory=False)
            df_pnl['time'] = pd.to_datetime(df_pnl['time'], errors='coerce')
            cutoff = pd.Timestamp.now() - pd.Timedelta(hours=48)
            mask_pnl = (
                (df_pnl['symbol'].astype(str).str.upper() == symbol_u)
                & (df_pnl['time'] >= cutoff)
                & (df_pnl['incomeType'] == 'REALIZED_PNL')
            )
            pnl_val = pd.to_numeric(df_pnl.loc[mask_pnl, 'income'], errors='coerce').sum()

        # Duración
        duration_str = ""
        if entry_time is not None and not pd.isna(entry_time):
            now_utc = pd.Timestamp.now(tz='UTC')
            delta = now_utc - entry_time
            total_min = int(delta.total_seconds() / 60)
            if total_min >= 60:
                hours = total_min // 60
                mins = total_min % 60
                duration_str = f"{hours}h {mins}m"
            else:
                duration_str = f"{total_min}m"

        # Emoji resultado
        if 'sl' in close_reason.lower() or 'stop' in close_reason.lower():
            result_emoji = "🔴"
            result_label = "SL"
        else:
            result_emoji = "✅"
            result_label = close_reason.upper().replace("_FILLED", "").replace("_", "")

        pnl_emoji = "🟢" if pnl_val >= 0 else "🔴"
        side_emoji = "🟢" if pside_u == "LONG" else "🔴"

        lines = [
            f"{'━' * 14}",
            f"📋 *TRADE CERRADO*",
            f"{'━' * 14}",
            f"{side_emoji} *{pair_name}* {pside_u}",
            "",
        ]
        if entry_price is not None and not pd.isna(entry_price):
            lines.append(f"Entrada  `{entry_price:.4f}`")
        lines.append(f"PnL       {pnl_emoji}`{pnl_val:+.2f} USD`")
        if duration_str:
            lines.append(f"Duración  `{duration_str}`")
        lines.append(f"Resultado  {result_emoji} {result_label}")

        msg = "\n".join(lines)
        bot_send_text(msg)

        # Log a CSV
        try:
            row = {
                'ts_utc': _utc_now_iso(),
                'symbol': symbol_u,
                'position_side': pside_u,
                'close_reason': close_reason,
                'pnl': round(pnl_val, 4),
                'duration_min': int(delta.total_seconds() / 60) if entry_time is not None and not pd.isna(entry_time) else 0,
                'entry_price': entry_price if entry_price is not None and not pd.isna(entry_price) else '',
            }
            df_log = pd.DataFrame([row])
            exists = os.path.exists(_TRADE_CLOSED_CSV) and os.path.getsize(_TRADE_CLOSED_CSV) > 0
            df_log.to_csv(_TRADE_CLOSED_CSV, mode='a', header=not exists, index=False)
        except Exception:
            pass

    except Exception as e:
        print(f"Error en _record_trade_closed: {e}")


# ---------------------------------------------------------------------------
#  Mensaje: Resumen diario (se llama una vez al día)
# ---------------------------------------------------------------------------
def daily_summary():
    """Envía resumen del día: trades cerrados, PnL, win rate."""
    try:
        pnl_csv = './archivos/PnL.csv'
        if not os.path.exists(pnl_csv):
            return

        df = pd.read_csv(pnl_csv, low_memory=False)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')

        today = pd.Timestamp.now().normalize()
        mask = (
            (df['time'] >= today)
            & (df['incomeType'] == 'REALIZED_PNL')
        )
        df_today = df[mask].copy()

        if df_today.empty:
            msg = "\n".join([
                f"{'━' * 14}",
                "📅 *RESUMEN DEL DÍA*",
                f"{'━' * 14}",
                "",
                "_Sin trades cerrados hoy_",
            ])
            bot_send_text(msg)
            return

        df_today['income'] = pd.to_numeric(df_today['income'], errors='coerce').fillna(0)

        # Agrupar por símbolo para contar trades
        by_sym = df_today.groupby('symbol')['income'].sum()
        total_pnl = by_sym.sum()
        wins = (by_sym > 0).sum()
        losses = (by_sym <= 0).sum()
        total_trades = len(by_sym)
        winrate = (wins / total_trades * 100) if total_trades > 0 else 0

        best_sym = by_sym.idxmax().replace('-USDT', '') if not by_sym.empty else "-"
        best_val = by_sym.max() if not by_sym.empty else 0
        worst_sym = by_sym.idxmin().replace('-USDT', '') if not by_sym.empty else "-"
        worst_val = by_sym.min() if not by_sym.empty else 0

        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

        lines = [
            f"{'���' * 14}",
            "📅 *RESUMEN DEL DÍA*",
            f"{'━' * 14}",
            "",
            f"Trades   `{total_trades}` ({wins}W / {losses}L)",
            f"PnL       {pnl_emoji}`{total_pnl:+.2f} USD`",
            f"Win rate  `{winrate:.0f}%`",
            "",
            f"Mejor    `{best_sym}` `{best_val:+.2f}`",
            f"Peor     `{worst_sym}` `{worst_val:+.2f}`",
        ]

        # Posiciones abiertas
        try:
            open_count = _count_open_positions()
            if open_count > 0:
                lines.append(f"\n📊 `{open_count}` posiciones abiertas")
        except Exception:
            pass

        msg = "\n".join(lines)
        bot_send_text(msg)

    except Exception as e:
        print(f"Error en daily_summary: {e}")


# ---------------------------------------------------------------------------
#  Mensaje: Posiciones abiertas (se llama cada 6h)
# ---------------------------------------------------------------------------
def _count_open_positions() -> int:
    """Cuenta posiciones abiertas consultando currencies activas."""
    count = 0
    try:
        currencies = pkg.price_bingx_5m.currencies_list()
        for curr in currencies:
            try:
                raw = pkg.bingx.perpetual_swap_positions(curr)
                data = json.loads(raw)
                if data.get('data'):
                    for pos in data['data']:
                        amt = float(pos.get('positionAmt', 0))
                        if abs(amt) > 0:
                            count += 1
            except Exception:
                continue
    except Exception:
        pass
    return count


def open_positions_alert():
    """Envía snapshot de posiciones abiertas con unrealized PnL."""
    try:
        currencies = pkg.price_bingx_5m.currencies_list()
        positions = []

        for curr in currencies:
            try:
                raw = pkg.bingx.perpetual_swap_positions(curr)
                data = json.loads(raw)
                if not data.get('data'):
                    continue
                for pos in data['data']:
                    amt = float(pos.get('positionAmt', 0))
                    if abs(amt) <= 0:
                        continue
                    pside = str(pos.get('positionSide', '')).upper()
                    unrealized = float(pos.get('unrealizedProfit', 0))
                    avg_price = float(pos.get('avgPrice', 0))
                    mark_price = float(pos.get('markPrice', avg_price))
                    pair_name = str(curr).replace('-USDT', '')

                    # Calcular % de ganancia
                    if avg_price > 0:
                        if pside == 'LONG':
                            pct = (mark_price / avg_price - 1) * 100
                        else:
                            pct = (1 - mark_price / avg_price) * 100
                    else:
                        pct = 0.0

                    positions.append({
                        'pair': pair_name,
                        'side': pside,
                        'pct': pct,
                        'unrealized': unrealized,
                    })
            except Exception:
                continue

        if not positions:
            return  # Sin posiciones abiertas, no enviar nada

        # Ordenar por unrealized PnL
        positions.sort(key=lambda x: x['unrealized'], reverse=True)

        total_unrealized = sum(p['unrealized'] for p in positions)
        total_emoji = "🟢" if total_unrealized >= 0 else "🔴"

        lines = [
            f"{'━' * 14}",
            "📊 *POSICIONES ABIERTAS*",
            f"{'━' * 14}",
            "",
        ]

        for p in positions:
            emoji = "🟢" if p['pct'] >= 0 else "🔴"
            side_short = "L" if p['side'] == 'LONG' else "S"
            lines.append(f"{emoji} `{p['pair']:>5}` {side_short}  `{p['pct']:+.2f}%`")

        lines.append("")
        lines.append(f"Total  {total_emoji}`{total_unrealized:+.2f} USD`")

        msg = "\n".join(lines)
        bot_send_text(msg)

    except Exception as e:
        print(f"Error en open_positions_alert: {e}")


#Funcion Enviar Mensajes
def bot_send_text(bot_message):
    text = str(bot_message or "")
    bot_token = str(getattr(pkg.credentials, "token", "") or "").strip()
    bot_chatID = str(getattr(pkg.credentials, "chatID", "") or "").strip()
    if not bot_token or not bot_chatID:
        print("Telegram no configurado: token/chat_id vacios para bot_send_text.")
        emit_lifecycle_event(
            "execution_quality_warning",
            "WARN",
            reason="telegram_missing_credentials",
            source="bot_send_text",
            detail="token_or_chat_id_empty",
        )
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": bot_chatID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        response = requests.post(url, data=payload, timeout=12)
        if response.ok:
            return True
        # Fallback sin parse_mode (por errores de markdown/formato).
        payload_plain = {
            "chat_id": bot_chatID,
            "text": text,
        }
        response_plain = requests.post(url, data=payload_plain, timeout=12)
        if response_plain.ok:
            return True
        detail = f"telegram_status={response_plain.status_code}:{(response_plain.text or '')[:180]}"
        print(f"Error enviando Telegram (plain fallback): {detail}")
        emit_lifecycle_event(
            "execution_quality_warning",
            "WARN",
            reason="telegram_send_failed",
            source="bot_send_text",
            detail=detail,
        )
        return False
    except Exception as exc:
        detail = str(exc)[:220]
        print(f"Error enviando Telegram: {detail}")
        emit_lifecycle_event(
            "execution_quality_warning",
            "WARN",
            reason="telegram_request_error",
            source="bot_send_text",
            detail=detail,
        )
        return False

def total_monkey():
    """
    Devuelve el balance de la cuenta.
    Maneja respuestas vacías o JSON mal formado para evitar que el bot se caiga.
    """
    raw = pkg.bingx.get_balance()

    # ── Validaciones defensivas ──────────────────────────────────────────
    if not raw or not raw.strip():
        print("Error: get_balance() devolvió cadena vacía.")
        return 0.0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error decodificando JSON: {e} → {raw[:120]!r}")
        return 0.0
    # ─────────────────────────────────────────────────────────────────────

    balance = 0.0
    try:
        balance = float(data['data']['balance']['balance'])
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error al obtener el balance: {e}")
        balance = 0.0  # Valor predeterminado en caso de error

    # Registrar histórico de balances
    datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_new = pd.DataFrame({'date': [datenow], 'balance': [balance]})

    file_path = './archivos/ganancias.csv'
    try:
        df_old = pd.read_csv(file_path)
        df_total = pd.concat([df_old, df_new]).tail(10000)
    except FileNotFoundError:
        df_total = df_new

    df_total.to_csv(file_path, index=False)

    return balance

def resultado_PnL():
    csv_path = './archivos/PnL.csv'
    try:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df_data = pd.read_csv(csv_path)
        else:
            df_data = pd.DataFrame()
    except Exception as exc:
        print(f"⚠️ Error leyendo PnL.csv: {exc}")
        df_data = pd.DataFrame()

    try:
        npl = pkg.bingx.hystory_PnL()
        npl = json.loads(npl)
    except Exception as exc:
        print(f"⚠️ Error obteniendo historial PnL: {exc}")
        return

    # Verificar si 'data' tiene datos
    if 'data' in npl and npl['data']:
        df = pd.DataFrame(npl['data'])

        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms')
        else:
            print("⚠️ Columna 'time' no presente en datos PnL. Se omite actualización.")
            return

        df_concat = pd.concat([df_data, df])
        df_unique = df_concat.drop_duplicates()
        df_limited = df_unique.tail(10000)
        df_limited.to_csv(csv_path, index=False)
    else:
        print("No hay datos nuevos para procesar en 'npl'.")



def monkey_result():
    # Obteniendo último resultado
    balance_actual = float(total_monkey())

    # Cargar los datos desde el archivo CSV en un DataFrame de pandas
    csv_path = './archivos/ganancias.csv'
    if not os.path.exists(csv_path):
        print("⚠️ ganancias.csv no existe. monkey_result retorna ceros.")
        return balance_actual, 0.0, 0.0, 0.0

    df = pd.read_csv(csv_path)
    if df.empty or 'date' not in df.columns or 'balance' not in df.columns:
        print("⚠️ ganancias.csv vacío o sin columnas esperadas.")
        return balance_actual, 0.0, 0.0, 0.0

    # Convertir la columna 'date' en formato datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filtrar los datos para obtener solo el día actual y la hora actual
    fecha_actual = datetime.now().date()
    hora_actual = datetime.now().hour
    df_dia_actual = df[df['date'].dt.date == fecha_actual]
    df_hora_actual = df_dia_actual[df_dia_actual['date'].dt.hour == hora_actual]

    # Diferencia del día
    if len(df_dia_actual) >= 1:
        balance_inicial_dia = df_dia_actual['balance'].iloc[0]
        balance_final_dia = df_dia_actual['balance'].iloc[-1]
        diferencia_dia = balance_final_dia - balance_inicial_dia
    else:
        diferencia_dia = 0.0

    # Diferencia de la hora
    if len(df_hora_actual) >= 1:
        balance_inicial_hora = df_hora_actual['balance'].iloc[0]
        balance_final_hora = df_hora_actual['balance'].iloc[-1]
        diferencia_hora = balance_final_hora - balance_inicial_hora
    else:
        diferencia_hora = 0.0

    # Calcular la fecha de una semana atrás
    fecha_semana_pasada = datetime.now().date() - timedelta(days=7)
    df_semana_pasada = df[df['date'].dt.date >= fecha_semana_pasada]

    # Diferencia de la semana
    if len(df_semana_pasada) >= 1:
        balance_inicial_semana = df_semana_pasada['balance'].iloc[0]
        balance_final_semana = df_semana_pasada['balance'].iloc[-1]
        diferencia_semana = balance_final_semana - balance_inicial_semana
    else:
        diferencia_semana = 0.0

    return balance_actual, diferencia_hora, diferencia_dia, diferencia_semana


# Obteniendo las Posiciones
def total_positions(symbol):
    # Obteniendo las posiciones desde la API o fuente de datos
    positions = pkg.bingx.perpetual_swap_positions(symbol)
    
    try:
        # Intenta decodificar el JSON
        positions = json.loads(positions)
    except json.JSONDecodeError:
        # Si hay un error en la decodificación, retorna None para todos los campos
        print("Error decodificando JSON. Posible respuesta vacía o mal formada.")
        return None, None, None, None, None

    # Verifica si 'data' está en la respuesta y que no está vacía
    if 'data' in positions and positions['data']:
        # Extrae los datos necesarios
        symbol = positions['data'][0]['symbol']
        positionSide = positions['data'][0]['positionSide']
        price = float(positions['data'][0]['avgPrice'])
        positionAmt = positions['data'][0]['positionAmt']
        unrealizedProfit = positions['data'][0]['unrealizedProfit']
        return symbol, positionSide, price, positionAmt, unrealizedProfit
    else:
        # Retorna None si 'data' no está presente o está vacía
        return None, None, None, None, None


#Obteniendo Ordenes Pendientes
def obteniendo_ordenes_pendientes():
    try:
        ordenes_raw = pkg.bingx.query_pending_orders()
        ordenes = json.loads(ordenes_raw)
    except json.JSONDecodeError as e:
        print("Error decodificando JSON:", e)
        return []

    data = ordenes.get('data', {})
    orders = data.get('orders', [])

    if not orders:
        df = pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
    else:
        df = pd.DataFrame(orders)

    try:
        prev_df = pd.read_csv(ORDER_PENDING_PREV_CSV) if os.path.exists(ORDER_PENDING_PREV_CSV) else pd.DataFrame(columns=['symbol','orderId','type','stopPrice','time'])
        prev_df = _normalize_orders_df(prev_df)
        df = _normalize_orders_df(df)
        csv_file = './archivos/order_id_register.csv'
        df.to_csv(csv_file, index=False)
        _log_pending_order_transitions(prev_df, df)
        df.to_csv(ORDER_PENDING_PREV_CSV, index=False)
    except Exception as e:
        print("Error al guardar en CSV:", e)

    return orders


def colocando_ordenes():
    pkg.monkey_bx.obteniendo_ordenes_pendientes()
    if not is_entry_hour_allowed_utc():
        return
    currencies = pkg.price_bingx_5m.currencies_list()
    # Cargar últimas señales de indicadores para mostrar TP escalonados en alertas
    latest_values = None
    try:
        _ind_path = './archivos/indicadores.csv'
        if os.path.exists(_ind_path):
            _dfi = pd.read_csv(_ind_path, low_memory=False)
            if 'date' in _dfi.columns:
                _dfi['date'] = pd.to_datetime(_dfi['date'])
            latest_values = _dfi.sort_values(['symbol','date']).groupby('symbol').last().reset_index()
    except Exception as _e:
        latest_values = None
    # --- Whitelist desde best_prod.json (si existe) ---
    best_prod_path = str(BEST_PROD_PATH)
    whitelist = None
    params_by_symbol = {}
    try:
        if os.path.exists(best_prod_path):
            with open(best_prod_path, 'r') as f:
                _prod = json.load(f)
            whitelist = set(str(x.get('symbol', '')).upper() for x in _prod if isinstance(x, dict))
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        whitelist = None
        params_by_symbol = {}
    if whitelist:
        currencies = [c for c in currencies if str(c).upper() in whitelist]
    # ---------------------------------------------------
    df_orders = _load_orders_register_df()

    # Actualizar cooldowns a partir de SL ejecutados
    sync_cooldowns_from_sl_fills()
    active_cooldowns = _load_active_cooldowns()
    now = datetime.utcnow()

    try:
        df_positions = pd.read_csv('./archivos/position_id_register.csv')
    except FileNotFoundError:
        df_positions = pd.DataFrame(columns=['symbol','tipo','counter'])

    # ── Position sizing: equal weight por par ──
    # Exposición total objetivo: 200% del capital (futuros con apalancamiento)
    # Cap individual: 50% para no concentrar en un solo par
    MAX_EXPOSURE_TOTAL = 2.0
    MAX_PER_TRADE = 0.50
    n_pairs = max(len(currencies), 1)
    peso_equal = min(MAX_EXPOSURE_TOTAL / n_pairs, MAX_PER_TRADE)

    # Obtener el total de fondos disponibles
    total_money = float(pkg.monkey_bx.total_monkey())
    capital_disponible = total_money

    # Lista para almacenar las monedas con señales activas
    active_currencies = []

    for currency in currencies:
        # Verificar si la moneda ya está en el DataFrame de órdenes o de posiciones
        if currency in df_orders['symbol'].values or currency in df_positions['symbol'].values:
            continue

        expiry = active_cooldowns.get(currency.upper())
        if expiry and now < expiry:
            continue

        try:
            # Obtener precio y tipo de alerta
            price_last, tipo = pkg.indicadores.ema_alert(currency)

            # Verificar si se obtuvo una alerta válida
            if "Alerta de LONG" not in str(tipo) and "Alerta de SHORT" not in str(tipo):
                continue  # No hay alerta para esta moneda

            # Asegurar que price_last es numérico
            try:
                price_last = float(price_last)
            except (TypeError, ValueError):
                continue

            # Añadir a la lista de monedas activas
            active_currencies.append({
                'symbol': currency,
                'tipo': tipo,
                'price_last': price_last,
                'peso': peso_equal,
            })

        except Exception as e:
            pass  # Manejo de excepciones

    # Si no hay monedas activas, terminar la función
    if not active_currencies:
        # Silenciado: print("No hay señales activas en este momento.")
        return

    # Asignar capital a cada moneda activa con equal weight
    total_capital_asignado = 0
    for item in active_currencies:
        trade = total_money * item['peso']
        # Verificar si el capital asignado excede el capital disponible
        if total_capital_asignado + trade > capital_disponible:
            trade = capital_disponible - total_capital_asignado
            if trade <= 0:
                print(f"No hay capital disponible para {item['symbol']}.")
                continue
        total_capital_asignado += trade
        item['trade'] = trade

    # Colocar las órdenes
    for item in active_currencies:
        if 'trade' not in item:
            continue  # Si no se asignó capital, pasar a la siguiente moneda

        currency = item['symbol']
        tipo = item['tipo']
        price_last = item['price_last']
        trade = item['trade']
        peso = item['peso']

        # Ajustar cantidad al step‑size permitido por el contrato
        step_size = _step_size_for(currency)
        raw_qty = trade / price_last
        currency_amount = math.floor(raw_qty / step_size) * step_size
        # Redondeo para evitar floats interminables
        decs = max(0, -int(round(math.log10(step_size))))
        currency_amount = round(currency_amount, decs)
        if currency_amount <= 0:
            continue

        # Determinar lado de la orden
        if "LONG" in str(tipo):
            order_side = "BUY"
            position_side = "LONG"
        elif "SHORT" in str(tipo):
            order_side = "SELL"
            position_side = "SHORT"
        else:
            continue  # Si no es ninguno, no colocar la orden

        # ---- Niveles TP/SL: usar indicadores.csv; fallback a best_prod.json ----
        tp_price, sl_price = get_last_take_profit_stop_loss(currency)
        if tp_price is None or sl_price is None:
            # Fallback: usa 'tp' de best_prod para el TP del mensaje; SL conservador ±0.5%
            p = params_by_symbol.get(str(currency).upper(), {})
            try:
                tp_pct = float(p.get('tp', 0.015))
            except Exception:
                tp_pct = 0.015
            if position_side == 'LONG':
                tp_price = price_last * (1.0 + tp_pct)
                sl_price = price_last * 0.995  # -0.5% como fallback textual
            else:  # SHORT
                tp_price = price_last * (1.0 - tp_pct)
                sl_price = price_last * 1.005  # +0.5% como fallback textual
        # -----------------------------------------------------------------------

        # Colocando la orden de entrada (benchmark: LIMIT + POST_ONLY; fallback opcional a MARKET).
        entry_mode = get_entry_mode()
        entry_ok = False
        entry_details = {}
        if entry_mode == "limit_post_only":
            entry_limit_price = _sanitize_entry_limit_price(
                price_last,
                currency,
                position_side,
                offset_bps=get_entry_limit_offset_bps(),
            )
            entry_kwargs = {}
            if is_entry_post_only_enabled():
                entry_kwargs["timeInForce"] = get_entry_time_in_force()
                entry_kwargs["postOnly"] = True
            entry_ok, entry_details = _post_with_retry(
                currency,
                currency_amount,
                entry_limit_price,
                0,
                position_side,
                "LIMIT",
                order_side,
                order_kwargs=entry_kwargs,
                intended_entry_price=price_last,
                emit_entry_lifecycle=True,
                return_details=True,
            )
            if (not entry_ok) and is_entry_market_fallback_on_error_enabled():
                emit_lifecycle_event(
                    "execution_quality_warning",
                    "WARN",
                    symbol=str(currency).upper(),
                    position_side=str(position_side).upper(),
                    side=str(order_side).upper(),
                    reason="entry_limit_post_only_failed_fallback_market",
                    source="colocando_ordenes_entry_fallback",
                )
                entry_ok, entry_details = _post_with_retry(
                    currency,
                    currency_amount,
                    0,
                    0,
                    position_side,
                    "MARKET",
                    order_side,
                    intended_entry_price=price_last,
                    emit_entry_lifecycle=True,
                    return_details=True,
                )
        else:
            entry_ok, entry_details = _post_with_retry(
                currency,
                currency_amount,
                0,
                0,
                position_side,
                "MARKET",
                order_side,
                intended_entry_price=price_last,
                emit_entry_lifecycle=True,
                return_details=True,
            )
        if not entry_ok:
            print(f"No se pudo abrir posición para {currency}; se omite configuración de TP/SL.")
            continue

        _append_entry_watch(
            symbol=currency,
            position_side=position_side,
            qty=_safe_float(currency_amount, 0.0),
            request_id=entry_details.get('request_id', ''),
            order_id=entry_details.get('order_id', ''),
            side=order_side,
            intended_entry_price=entry_details.get('intended_entry_price'),
            submitted_price=entry_details.get('submitted_price'),
            submit_time_utc=entry_details.get('submit_time_utc', ''),
            trade_capital=round(trade, 2),
            peso_pct=round(peso * 100, 1),
        )
        upsert_tp_state(
            currency,
            position_side,
            tp_mode=get_tp_mode(),
            tp_stage="none",
            break_even_state="inactive",
            tp_fill_confirmation_mode=get_tp_fill_confirmation_mode(),
        )

        # Guardando las posiciones
        nueva_fila = pd.DataFrame({
            'symbol': [currency],
            'tipo': [position_side],
            'counter': [0]
        })
        df_positions = pd.concat([df_positions, nueva_fila], ignore_index=True)

        # La alerta Telegram se envía unificada al confirmar el fill
        # en colocando_TK_SL (entry_order_filled).
        print(f"📡 Orden enviada: {currency} {position_side} @ {price_last}")

    # Guardando Posiciones fuera del bucle
    df_positions.to_csv('./archivos/position_id_register.csv', index=False)

def sync_cooldowns_from_sl_fills():
    try:
        if not os.path.exists(SL_WATCH_CSV):
            return
        df_watch = pd.read_csv(SL_WATCH_CSV)
    except Exception as e:
        print(f"No se pudo leer SL watch: {e}")
        return

    if df_watch.empty:
        return

    df_watch['symbol'] = df_watch['symbol'].astype(str).str.upper()
    df_watch['position_side'] = df_watch['position_side'].astype(str).str.upper()

    try:
        df_orders = pd.read_csv('./archivos/order_id_register.csv')
        df_orders = _normalize_orders_df(df_orders)
    except Exception:
        df_orders = pd.DataFrame(columns=['symbol','orderId','type','stopPrice'])

    pending = df_orders[df_orders['type'] == 'STOP_MARKET'].copy()
    pending['symbol'] = pending['symbol'].astype(str).str.upper()
    pending_ids = set(str(x) for x in pending.get('orderId', []))

    # Pre-cargar parámetros por símbolo para determinar duración del cooldown
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception:
        params_by_symbol = {}

    remaining_rows = []
    now = datetime.utcnow()

    for _, row in df_watch.iterrows():
        symbol = str(row.get('symbol', '')).upper()
        position_side = str(row.get('position_side', '')).upper()
        order_id = str(row.get('orderId', '')).strip()
        stop_price = None
        try:
            stop_price = float(row.get('stop_price'))
        except Exception:
            stop_price = None

        still_pending = False
        if order_id:
            still_pending = order_id in pending_ids
        else:
            try:
                subset = pending[pending['symbol'] == symbol]
                if not subset.empty and stop_price is not None and 'stopPrice' in subset.columns:
                    subset['stopPrice'] = pd.to_numeric(subset['stopPrice'], errors='coerce')
                    still_pending = any(abs((float(sp) - stop_price) / stop_price) < 5e-4 for sp in subset['stopPrice'].dropna())
            except Exception:
                still_pending = False

        if still_pending:
            remaining_rows.append(row)
            continue

        minutes = _cooldown_minutes_for_symbol(params_by_symbol, symbol, default=10)
        _write_cooldown(symbol, now, minutes)
        fill_ts_utc = _utc_now_iso()
        emit_lifecycle_event(
            "stop_loss_hit",
            "WARN",
            symbol=symbol,
            position_side=position_side,
            order_id=_norm_order_id(order_id),
            stop_price=stop_price if stop_price is not None else None,
            cooldown_min=minutes,
            source="inferred_sl_watch_disappeared",
        )
        append_execution_ledger_event(
            "stop_loss_hit",
            data_quality="inferred",
            source="sl_watch_disappeared_inference",
            ts_utc=fill_ts_utc,
            order_id=_norm_order_id(order_id),
            symbol=symbol,
            position_side=position_side,
            order_type="STOP_MARKET",
            stop_price=stop_price if stop_price is not None else None,
            submit_time_utc=str(row.get('ts', '')).strip(),
            fill_time_utc=fill_ts_utc,
            close_reason="stop_loss",
            partial_fill_status="unknown",
            notes=f"cooldown_applied_min={minutes}",
        )
        try:
            clear_tp_state(symbol, position_side)
        except Exception:
            pass
        # Alerta de trade cerrado
        try:
            _record_trade_closed(symbol, position_side, "stop_loss")
        except Exception:
            pass

    if remaining_rows:
        pd.DataFrame(remaining_rows).to_csv(SL_WATCH_CSV, index=False)
    else:
        try:
            os.remove(SL_WATCH_CSV)
        except Exception:
            pass


def colocando_TK_SL():
    # Obteniendo posiciones sin SL o TP (arranque en frío tolerante)
    try:
        df_posiciones = pd.read_csv('./archivos/position_id_register.csv')
    except FileNotFoundError:
        df_posiciones = pd.DataFrame(columns=['symbol','tipo','counter'])
    # Asegurarse de que exista la columna 'counter'
    if 'counter' not in df_posiciones.columns:
        df_posiciones['counter'] = 0
    if not df_posiciones.empty:
        df_posiciones['counter'] += 1

    df_ordenes = _load_orders_register_df()
    df_posiciones = _bootstrap_position_queue(df_posiciones, df_ordenes)

    # Leer los últimos valores de indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    # Limpiar espacios en los nombres de columnas y en los símbolos
    df_indicadores.columns = df_indicadores.columns.str.strip()
    if 'symbol' in df_indicadores.columns:
        df_indicadores['symbol'] = df_indicadores['symbol'].str.strip().str.upper()
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Cargar parametros por símbolo (para ajustar TP1) desde pkg/best_prod.json
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        params_by_symbol = {}

    tp_mode_runtime = get_tp_mode()
    tp_fill_mode = get_tp_fill_confirmation_mode()
    tp_splits_runtime = _runtime_tp_splits()

    # Agrega un pequeño delay antes de buscar la posición en BingX para dar tiempo a que la posición MARKET aparezca
    time.sleep(2)
    for index, row in df_posiciones.iterrows():
        symbol = str(row['symbol']).upper()
        counter = row['counter']

        # Verificar si ya existen órdenes SL y TP pendientes para este símbolo
        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
        position_side_hint = str(row.get('tipo', '')).upper().strip()
        sl_exists = not symbol_orders[symbol_orders['type'] == 'STOP_MARKET'].empty
        tp_exists = not _extract_tp_orders(symbol_orders, position_side_hint, tp_mode_runtime).empty

        # Si ambos existen, ya está protegido: eliminamos la entrada y seguimos
        if sl_exists and tp_exists:
            df_posiciones.drop(index, inplace=True)
            continue

        # Verificar si se debe cancelar la orden después de cierto tiempo
        if counter >= 20:
            try:
                mask = (df_ordenes['symbol'] == symbol) if 'symbol' in df_ordenes.columns else pd.Series([], dtype=bool)
                pair_name = symbol.replace('-USDT', '')
                orderId = None
                canceled = False
                if mask.any():
                    df_sym = df_ordenes.loc[mask]
                    if 'orderId' in df_sym.columns and not df_sym['orderId'].isna().all():
                        orderId = df_sym['orderId'].iloc[0]
                        try:
                            pkg.bingx.cancel_order(symbol, orderId)
                            canceled = True
                        except Exception as ce:
                            print(f"Error al cancelar la orden para {symbol}: {ce}")
                emoji = "⛔" if canceled else "⏳"
                status = "cancelada" if canceled else "expirada"
                msg = f"{emoji} *{pair_name}* — Orden {status}\n_No se ejecutó a tiempo_"
                # Retirar la posición de la cola de protección en cualquier caso
                df_posiciones.drop(index, inplace=True)
                df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)
                print(msg)
                try:
                    pkg.monkey_bx.bot_send_text(msg)
                except Exception:
                    pass
                emit_lifecycle_event(
                    "entry_order_canceled_or_expired",
                    "WARN",
                    symbol=str(symbol).upper(),
                    reason="protection_timeout",
                    source="colocando_TK_SL_counter",
                    detail=str(msg)[:220],
                )
                append_execution_ledger_event(
                    "entry_order_canceled_or_expired",
                    data_quality="inferred",
                    source="protection_timeout_counter",
                    order_id=_norm_order_id(orderId),
                    symbol=str(symbol).upper(),
                    cancel_reason="protection_timeout",
                    notes=str(msg)[:220],
                )
                continue  # Evita múltiples mensajes y cancelaciones para la misma orden
            except Exception as e:
                print(f"Error al manejar timeout para {symbol}: {e}")
                # Aun con error, sacar de la cola para no ciclar
                try:
                    df_posiciones.drop(index, inplace=True)
                    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)
                except Exception:
                    pass
                continue

        # Obteniendo el valor de las posiciones reales
        try:
            # Obtener los detalles de la posición actual
            result = total_positions(symbol)
            if result[0] is None:
                print(f"Posición para {symbol} aún no aparece en el exchange. Reintentando en el próximo ciclo.")
                continue

            symbol_result, positionSide, price, positionAmt, unrealizedProfit = result
            tp_mode_effective = tp_mode_runtime
            tp_mode_detect = "partial_limit_tp" if tp_mode_runtime == "partial_limit_tp" else tp_mode_runtime
            try:
                upsert_tp_state(
                    symbol,
                    positionSide,
                    tp_mode=tp_mode_runtime,
                    tp_fill_confirmation_mode=tp_fill_mode,
                )
                tp_state_status = get_tp_state_persist_status()
                if not bool(tp_state_status.get("ok", True)):
                    err_detail = str(tp_state_status.get("error", "tp_state_persist_unknown_error"))
                    warned = _emit_runtime_storage_warning(
                        symbol=symbol,
                        position_side=positionSide,
                        source="colocando_TK_SL_state_upsert",
                        reason="tp_state_persist_unhealthy",
                        detail=err_detail,
                        severity="CRITICAL",
                    )
                    if tp_mode_runtime == "partial_limit_tp":
                        tp_mode_effective = "legacy_market_tp"
                        if warned:
                            emit_lifecycle_event(
                                "legacy_fallback_used",
                                "WARN",
                                symbol=str(symbol).upper(),
                                position_side=str(positionSide).upper(),
                                reason="tp_state_persist_unhealthy",
                                source="colocando_TK_SL_state_guard",
                            )
                            append_execution_ledger_event(
                                "legacy_fallback_used",
                                data_quality="inferred",
                                source="colocando_TK_SL_state_guard",
                                symbol=str(symbol).upper(),
                                position_side=str(positionSide).upper(),
                                close_reason="tp_state_persist_unhealthy",
                                notes="forced_legacy_market_tp_for_symbol_cycle",
                            )
            except Exception as state_exc:
                warned = _emit_runtime_storage_warning(
                    symbol=symbol,
                    position_side=positionSide,
                    source="colocando_TK_SL_state_upsert_exception",
                    reason="tp_state_upsert_exception",
                    detail=str(state_exc),
                    severity="CRITICAL",
                )
                if tp_mode_runtime == "partial_limit_tp":
                    tp_mode_effective = "legacy_market_tp"
                    if warned:
                        emit_lifecycle_event(
                            "legacy_fallback_used",
                            "WARN",
                            symbol=str(symbol).upper(),
                            position_side=str(positionSide).upper(),
                            reason="tp_state_upsert_exception",
                            source="colocando_TK_SL_state_guard",
                        )
                        append_execution_ledger_event(
                            "legacy_fallback_used",
                            data_quality="inferred",
                            source="colocando_TK_SL_state_guard",
                            symbol=str(symbol).upper(),
                            position_side=str(positionSide).upper(),
                            close_reason="tp_state_upsert_exception",
                            notes=str(state_exc)[:220],
                        )
            if is_sl_guard_active(symbol, positionSide):
                continue
            market_ref = _last_traded_price(symbol)
            if market_ref is None or market_ref <= 0:
                try:
                    market_ref = float(price)
                except Exception:
                    market_ref = None

            try:
                position_qty = abs(float(positionAmt))
            except Exception:
                position_qty = 0.0

            watch_hit = _consume_entry_watch(symbol, positionSide)
            _send_fill_alert = False
            _fill_trade_capital = 0.0
            _fill_peso_pct = 0.0
            if watch_hit is not None:
                _send_fill_alert = True
                _fill_trade_capital = _safe_float(watch_hit.get("trade_capital"), 0.0)
                _fill_peso_pct = _safe_float(watch_hit.get("peso_pct"), 0.0)
                fill_ts_utc = _utc_now_iso()
                submit_ts_utc = str(watch_hit.get("submit_time_utc", "") or watch_hit.get("ts_utc", "")).strip()
                watch_side = str(watch_hit.get("side", "")).upper().strip()
                if not watch_side:
                    watch_side = "BUY" if str(positionSide).upper() == "LONG" else "SELL"
                emit_lifecycle_event(
                    "entry_order_filled",
                    "INFO",
                    symbol=str(symbol).upper(),
                    position_side=str(positionSide).upper(),
                    qty=position_qty,
                    avg_price=_safe_float(price, 0.0),
                    source="inferred_position_seen",
                    request_id=watch_hit.get("request_id", ""),
                )
                append_execution_ledger_event(
                    "entry_order_filled",
                    data_quality="inferred",
                    source="position_seen_inference",
                    ts_utc=fill_ts_utc,
                    request_id=str(watch_hit.get("request_id", "")).strip(),
                    order_id=_norm_order_id(watch_hit.get("order_id")),
                    symbol=str(symbol).upper(),
                    side=watch_side,
                    position_side=str(positionSide).upper(),
                    order_type="MARKET",
                    intended_entry_price=_safe_float_or_none(watch_hit.get("intended_entry_price")),
                    submitted_price=_safe_float_or_none(watch_hit.get("submitted_price")),
                    actual_fill_price=_safe_float_or_none(price),
                    submit_qty=_safe_float_or_none(watch_hit.get("qty")),
                    fill_qty=position_qty,
                    submit_time_utc=submit_ts_utc,
                    fill_time_utc=fill_ts_utc,
                    partial_fill_status="unknown",
                    notes="fill de entrada inferido por aparicion de posicion",
                )

            if positionSide == 'LONG':
                # Niveles TP (escalonados si existen) y SL desde indicadores
                desired_tps, sl_level = extract_tp_sl_from_latest(latest_values, symbol, 'LONG')
                # Fallbacks de emergencia si no hay datos
                if not desired_tps:
                    desired_tps = [price * 1.01]
                if sl_level is None:
                    sl_level = price * 0.995

                # Ajuste opcional de TP1 desde best_prod.json (más cerca del precio de entrada)
                try:
                    p = params_by_symbol.get(str(symbol).upper(), {})
                    tp1_factor = p.get('tp1_factor', None)
                    tp1_pct_override = p.get('tp1_pct_override', None)
                    tp1_factor = float(tp1_factor) if tp1_factor is not None else None
                    tp1_pct_override = float(tp1_pct_override) if tp1_pct_override is not None else None
                except Exception:
                    tp1_factor = None
                    tp1_pct_override = None

                if desired_tps:
                    if tp1_pct_override is not None and tp1_pct_override > 0:
                        desired_tps[0] = float(price) * (1.0 + tp1_pct_override)
                    elif tp1_factor is not None and 0.0 < tp1_factor < 1.0:
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) + (base - float(price)) * tp1_factor
                    else:
                        # Fallback: TP1 más alcanzable por defecto (70% del camino)
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) + (base - float(price)) * 0.70

                # Alerta unificada de operación confirmada
                if _send_fill_alert:
                    try:
                        _alert = _build_fill_alert(
                            symbol, 'LONG', float(price),
                            desired_tps, sl_level,
                            _fill_trade_capital, _fill_peso_pct,
                        )
                        pkg.monkey_bx.bot_send_text(_alert)
                    except Exception as _ae:
                        print(f"Error enviando alerta fill: {_ae}")
                    _send_fill_alert = False

                # ¿Qué órdenes existen ya?
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = _extract_tp_orders(symbol_orders, "LONG", tp_mode_detect)
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                tick = _tick_size_for(symbol)
                step_sz = _step_size_for(symbol)
                existing_tp_prices = _extract_tp_price_set(existing_tp, symbol)

                # Colocar SL si falta
                exito_sl = not existing_sl.empty
                if not exito_sl:
                    sl_px = _sanitize_trigger_price(float(sl_level), symbol, "LONG", "STOP_MARKET", market_ref)
                    exito_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "LONG", "STOP_MARKET", "SELL")
                    if exito_sl:
                        time.sleep(1)
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'LONG', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'LONG', None)

                # Cantidades parciales para TPs
                try:
                    pos_qty = abs(float(positionAmt))
                except Exception:
                    pos_qty = 0.0
                splits = tp_splits_runtime if len(desired_tps) >= 3 else (1.0,)
                split_qtys = _split_position_qtys(pos_qty, splits, step_sz)

                # Colocar TP(s) faltantes
                placed_all_tps = True
                if tp_mode_effective == "partial_limit_tp":
                    st_curr = get_tp_state(symbol, "LONG")
                    stage_idx_target = _next_tp_idx_from_stage(st_curr.get("tp_stage", "none"))
                    stage_price_ref = None

                    if stage_idx_target is not None and desired_tps:
                        price_src_idx = min(stage_idx_target - 1, max(len(desired_tps) - 1, 0))
                        stage_price_ref = _sanitize_tp_limit_price(
                            float(desired_tps[price_src_idx]),
                            symbol,
                            "LONG",
                            market_ref,
                            offset_bps=get_tp_limit_offset_bps(),
                        )
                        exists = any(abs((stage_price_ref - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                        if not exists:
                            stage_qty, qty_reason = _compute_partial_limit_stage_qty(
                                position_qty_now=pos_qty,
                                step_sz=step_sz,
                                splits=splits,
                                state=st_curr,
                                stage_idx=stage_idx_target,
                            )
                            if stage_qty <= 0:
                                _emit_tp_failed(
                                    tp_idx=stage_idx_target,
                                    symbol=symbol,
                                    position_side="LONG",
                                    reason=f"{_tp_stage_name(stage_idx_target)}_qty_invalid",
                                    detail=str(qty_reason),
                                    data_quality="inferred",
                                    source="colocando_TK_SL",
                                    tp_price=stage_price_ref,
                                    tp_qty=stage_qty,
                                )
                                placed_all_tps = False
                            else:
                                ok, tp_details = _post_with_retry(
                                    symbol,
                                    stage_qty,
                                    stage_price_ref,
                                    0,
                                    "LONG",
                                    "LIMIT",
                                    "SELL",
                                    order_kwargs=_tp_limit_order_kwargs(symbol, "LONG", tp_idx=stage_idx_target),
                                    return_details=True,
                                )
                                if ok:
                                    time.sleep(0.3)
                                    existing_tp_prices.add(stage_price_ref)
                                    set_tp_submitted(
                                        symbol,
                                        "LONG",
                                        tp_idx=stage_idx_target,
                                        order_id=tp_details.get("order_id", ""),
                                        qty=stage_qty,
                                        price=stage_price_ref,
                                        submit_position_qty=pos_qty,
                                        tp_mode=tp_mode_effective,
                                        fill_confirmation_mode=tp_fill_mode,
                                    )
                                    emit_lifecycle_event(
                                        f"tp{stage_idx_target}_submitted",
                                        "INFO",
                                        symbol=str(symbol).upper(),
                                        position_side="LONG",
                                        qty=stage_qty,
                                        tp_price=stage_price_ref,
                                        order_id=tp_details.get("order_id", ""),
                                        source="colocando_TK_SL",
                                    )
                                    append_execution_ledger_event(
                                        f"tp{stage_idx_target}_submitted",
                                        data_quality="actual",
                                        source="colocando_TK_SL_submit",
                                        request_id=tp_details.get("request_id", ""),
                                        order_id=tp_details.get("order_id", ""),
                                        symbol=str(symbol).upper(),
                                        side="SELL",
                                        position_side="LONG",
                                        order_type="LIMIT",
                                        submitted_price=stage_price_ref,
                                        stop_price=None,
                                        submit_qty=stage_qty,
                                        submit_time_utc=tp_details.get("submit_time_utc", ""),
                                        partial_fill_status="unknown",
                                    )
                                else:
                                    _emit_tp_failed(
                                        tp_idx=stage_idx_target,
                                        symbol=symbol,
                                        position_side="LONG",
                                        reason=f"{_tp_stage_name(stage_idx_target)}_submit_failed",
                                        detail="limit_submit_rejected_or_exception",
                                        order_id=tp_details.get("order_id", "") if isinstance(tp_details, dict) else "",
                                        data_quality="actual",
                                        source="colocando_TK_SL",
                                        tp_price=stage_price_ref,
                                        tp_qty=stage_qty,
                                    )
                                    if get_tp_legacy_fallback_on_error():
                                        ok_fb = _submit_tp_legacy_fallback(
                                            tp_idx=stage_idx_target,
                                            symbol=symbol,
                                            position_side="LONG",
                                            market_ref=market_ref,
                                            tp_price=stage_price_ref,
                                            tp_qty=stage_qty,
                                            tp_fill_mode=tp_fill_mode,
                                            reason=f"{_tp_stage_name(stage_idx_target)}_limit_submit_failed",
                                        )
                                        if not ok_fb:
                                            placed_all_tps = False
                                    else:
                                        placed_all_tps = False
                    elif stage_idx_target is not None and not desired_tps:
                        placed_all_tps = False

                    # Revalidar si la etapa objetivo esta viva tras submit.
                    try:
                        _orders = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                        existing_tp = _extract_tp_orders(symbol_orders, "LONG", tp_mode_detect)
                        st_new = get_tp_state(symbol, "LONG")

                        if stage_idx_target is None:
                            placed_all_tps = True
                        else:
                            stage_live = False
                            oid_target = _stage_order_id_from_state(st_new, stage_idx_target)
                            if oid_target and "orderId" in existing_tp.columns:
                                ids = existing_tp["orderId"].astype(str).str.strip().tolist()
                                stage_live = oid_target in ids
                            if not stage_live and stage_price_ref is not None:
                                existing_tp_prices = _extract_tp_price_set(existing_tp, symbol)
                                stage_live = any(abs((stage_price_ref - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                            # Si el estado ya avanzo (ej. tp1_filled -> tp2), no bloquear.
                            next_idx_after = _next_tp_idx_from_stage(st_new.get("tp_stage", "none"))
                            placed_all_tps = stage_live or (next_idx_after != stage_idx_target)
                    except Exception:
                        pass
                else:
                    target_tp_count = len(split_qtys)
                    for idx, tp_px in enumerate(desired_tps[:target_tp_count]):
                        tp_idx = idx + 1
                        tp_px = _sanitize_trigger_price(float(tp_px), symbol, "LONG", "TAKE_PROFIT_MARKET", market_ref)
                        exists = any(abs((tp_px - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                        if exists:
                            continue
                        tp_qty = split_qtys[idx]
                        if tp_qty <= 0:
                            continue
                        ok, tp_details = _post_with_retry(
                            symbol,
                            tp_qty,
                            0,
                            tp_px,
                            "LONG",
                            "TAKE_PROFIT_MARKET",
                            "SELL",
                            return_details=True,
                        )
                        if ok:
                            time.sleep(0.3)
                            existing_tp_prices.add(tp_px)
                            set_tp_submitted(
                                symbol,
                                "LONG",
                                tp_idx=tp_idx,
                                order_id=tp_details.get("order_id", ""),
                                qty=tp_qty,
                                price=tp_px,
                                submit_position_qty=pos_qty,
                                tp_mode=tp_mode_effective,
                                fill_confirmation_mode=tp_fill_mode,
                            )
                            emit_lifecycle_event(
                                f"tp{tp_idx}_submitted",
                                "INFO",
                                symbol=str(symbol).upper(),
                                position_side="LONG",
                                qty=tp_qty,
                                tp_price=tp_px,
                                order_id=tp_details.get("order_id", ""),
                                source="colocando_TK_SL",
                            )
                            append_execution_ledger_event(
                                f"tp{tp_idx}_submitted",
                                data_quality="actual",
                                source="colocando_TK_SL_submit",
                                request_id=tp_details.get("request_id", ""),
                                order_id=tp_details.get("order_id", ""),
                                symbol=str(symbol).upper(),
                                side="SELL",
                                position_side="LONG",
                                order_type="TAKE_PROFIT_MARKET",
                                submitted_price=0,
                                stop_price=tp_px,
                                submit_qty=tp_qty,
                                submit_time_utc=tp_details.get("submit_time_utc", ""),
                                partial_fill_status="unknown",
                            )
                        else:
                            placed_all_tps = False

                # Si SL existe y todos los TP están listos, retirar de la cola
                if exito_sl and placed_all_tps:
                    df_posiciones.drop(index, inplace=True)

                # Fallbacks adicionales: garantizar al menos una protección.
                # Refrescar snapshot para evitar duplicar envíos con estado stale.
                try:
                    _ = obteniendo_ordenes_pendientes()
                except Exception:
                    pass
                df_ordenes = _load_orders_register_df()
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = _extract_tp_orders(symbol_orders, "LONG", tp_mode_detect)
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                if existing_sl.empty:
                    sl_px = _sanitize_trigger_price(float(sl_level), symbol, "LONG", "STOP_MARKET", market_ref)
                    ok_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "LONG", "STOP_MARKET", "SELL")
                    time.sleep(0.3)
                    if ok_sl:
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'LONG', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'LONG', None)
                if tp_mode_effective != "partial_limit_tp" and existing_tp.empty and desired_tps:
                    tp1_px = _sanitize_trigger_price(float(desired_tps[0]), symbol, "LONG", "TAKE_PROFIT_MARKET", market_ref)
                    try:
                        pos_qty = abs(float(positionAmt))
                    except Exception:
                        pos_qty = 0.0
                    tp_qty = _round_step(pos_qty, step_sz)
                    if tp_qty > 0:
                        ok, tp_details = _post_with_retry(
                            symbol,
                            tp_qty,
                            0,
                            tp1_px,
                            "LONG",
                            "TAKE_PROFIT_MARKET",
                            "SELL",
                            return_details=True,
                        )
                        if ok:
                            set_tp_submitted(
                                symbol,
                                "LONG",
                                tp_idx=1,
                                order_id=tp_details.get("order_id", ""),
                                qty=tp_qty,
                                price=tp1_px,
                                tp_mode=tp_mode_effective,
                                fill_confirmation_mode=tp_fill_mode,
                            )
                        time.sleep(0.3)

            elif positionSide == 'SHORT':
                # Niveles TP (escalonados si existen) y SL desde indicadores
                desired_tps, sl_level = extract_tp_sl_from_latest(latest_values, symbol, 'SHORT')
                if not desired_tps:
                    desired_tps = [price * 0.99]
                if sl_level is None:
                    sl_level = price * 1.005

                # Ajuste opcional de TP1 desde best_prod.json para SHORT
                try:
                    p = params_by_symbol.get(str(symbol).upper(), {})
                    tp1_factor = p.get('tp1_factor', None)
                    tp1_pct_override = p.get('tp1_pct_override', None)
                    tp1_factor = float(tp1_factor) if tp1_factor is not None else None
                    tp1_pct_override = float(tp1_pct_override) if tp1_pct_override is not None else None
                except Exception:
                    tp1_factor = None
                    tp1_pct_override = None

                if desired_tps:
                    if tp1_pct_override is not None and tp1_pct_override > 0:
                        desired_tps[0] = float(price) * (1.0 - tp1_pct_override)
                    elif tp1_factor is not None and 0.0 < tp1_factor < 1.0:
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) - (float(price) - base) * tp1_factor
                    else:
                        # Fallback: TP1 más alcanzable por defecto (70% del camino)
                        base = float(desired_tps[0])
                        desired_tps[0] = float(price) - (float(price) - base) * 0.70

                # Alerta unificada de operación confirmada
                if _send_fill_alert:
                    try:
                        _alert = _build_fill_alert(
                            symbol, 'SHORT', float(price),
                            desired_tps, sl_level,
                            _fill_trade_capital, _fill_peso_pct,
                        )
                        pkg.monkey_bx.bot_send_text(_alert)
                    except Exception as _ae:
                        print(f"Error enviando alerta fill: {_ae}")
                    _send_fill_alert = False

                # ¿Qué órdenes existen ya?
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = _extract_tp_orders(symbol_orders, "SHORT", tp_mode_detect)
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                tick = _tick_size_for(symbol)
                step_sz = _step_size_for(symbol)
                existing_tp_prices = _extract_tp_price_set(existing_tp, symbol)

                # SL si falta
                sl_px = _sanitize_trigger_price(float(sl_level), symbol, "SHORT", "STOP_MARKET", market_ref)
                exito_sl = not existing_sl.empty
                if not exito_sl:
                    exito_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "SHORT", "STOP_MARKET", "BUY")
                    if exito_sl:
                        time.sleep(1)
                # Registrar SL en watch para detectar fill y disparar cooldown
                try:
                    _ = obteniendo_ordenes_pendientes()
                    df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                    m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                    order_id = None
                    if not m.empty:
                        if 'stopPrice' in m.columns:
                            m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                            m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                    _append_sl_watch(symbol, float(sl_px), 'SHORT', order_id)
                except Exception:
                    _append_sl_watch(symbol, float(sl_px), 'SHORT', None)

                # Cantidades parciales
                try:
                    pos_qty = abs(float(positionAmt))
                except Exception:
                    pos_qty = 0.0
                splits = tp_splits_runtime if len(desired_tps) >= 3 else (1.0,)
                split_qtys = _split_position_qtys(pos_qty, splits, step_sz)

                # Colocar TPs faltantes
                placed_all_tps = True
                if tp_mode_effective == "partial_limit_tp":
                    st_curr = get_tp_state(symbol, "SHORT")
                    stage_idx_target = _next_tp_idx_from_stage(st_curr.get("tp_stage", "none"))
                    stage_price_ref = None

                    if stage_idx_target is not None and desired_tps:
                        price_src_idx = min(stage_idx_target - 1, max(len(desired_tps) - 1, 0))
                        stage_price_ref = _sanitize_tp_limit_price(
                            float(desired_tps[price_src_idx]),
                            symbol,
                            "SHORT",
                            market_ref,
                            offset_bps=get_tp_limit_offset_bps(),
                        )
                        exists = any(abs((stage_price_ref - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                        if not exists:
                            stage_qty, qty_reason = _compute_partial_limit_stage_qty(
                                position_qty_now=pos_qty,
                                step_sz=step_sz,
                                splits=splits,
                                state=st_curr,
                                stage_idx=stage_idx_target,
                            )
                            if stage_qty <= 0:
                                _emit_tp_failed(
                                    tp_idx=stage_idx_target,
                                    symbol=symbol,
                                    position_side="SHORT",
                                    reason=f"{_tp_stage_name(stage_idx_target)}_qty_invalid",
                                    detail=str(qty_reason),
                                    data_quality="inferred",
                                    source="colocando_TK_SL",
                                    tp_price=stage_price_ref,
                                    tp_qty=stage_qty,
                                )
                                placed_all_tps = False
                            else:
                                ok, tp_details = _post_with_retry(
                                    symbol,
                                    stage_qty,
                                    stage_price_ref,
                                    0,
                                    "SHORT",
                                    "LIMIT",
                                    "BUY",
                                    order_kwargs=_tp_limit_order_kwargs(symbol, "SHORT", tp_idx=stage_idx_target),
                                    return_details=True,
                                )
                                if ok:
                                    time.sleep(0.3)
                                    existing_tp_prices.add(stage_price_ref)
                                    set_tp_submitted(
                                        symbol,
                                        "SHORT",
                                        tp_idx=stage_idx_target,
                                        order_id=tp_details.get("order_id", ""),
                                        qty=stage_qty,
                                        price=stage_price_ref,
                                        submit_position_qty=pos_qty,
                                        tp_mode=tp_mode_effective,
                                        fill_confirmation_mode=tp_fill_mode,
                                    )
                                    emit_lifecycle_event(
                                        f"tp{stage_idx_target}_submitted",
                                        "INFO",
                                        symbol=str(symbol).upper(),
                                        position_side="SHORT",
                                        qty=stage_qty,
                                        tp_price=stage_price_ref,
                                        order_id=tp_details.get("order_id", ""),
                                        source="colocando_TK_SL",
                                    )
                                    append_execution_ledger_event(
                                        f"tp{stage_idx_target}_submitted",
                                        data_quality="actual",
                                        source="colocando_TK_SL_submit",
                                        request_id=tp_details.get("request_id", ""),
                                        order_id=tp_details.get("order_id", ""),
                                        symbol=str(symbol).upper(),
                                        side="BUY",
                                        position_side="SHORT",
                                        order_type="LIMIT",
                                        submitted_price=stage_price_ref,
                                        stop_price=None,
                                        submit_qty=stage_qty,
                                        submit_time_utc=tp_details.get("submit_time_utc", ""),
                                        partial_fill_status="unknown",
                                    )
                                else:
                                    _emit_tp_failed(
                                        tp_idx=stage_idx_target,
                                        symbol=symbol,
                                        position_side="SHORT",
                                        reason=f"{_tp_stage_name(stage_idx_target)}_submit_failed",
                                        detail="limit_submit_rejected_or_exception",
                                        order_id=tp_details.get("order_id", "") if isinstance(tp_details, dict) else "",
                                        data_quality="actual",
                                        source="colocando_TK_SL",
                                        tp_price=stage_price_ref,
                                        tp_qty=stage_qty,
                                    )
                                    if get_tp_legacy_fallback_on_error():
                                        ok_fb = _submit_tp_legacy_fallback(
                                            tp_idx=stage_idx_target,
                                            symbol=symbol,
                                            position_side="SHORT",
                                            market_ref=market_ref,
                                            tp_price=stage_price_ref,
                                            tp_qty=stage_qty,
                                            tp_fill_mode=tp_fill_mode,
                                            reason=f"{_tp_stage_name(stage_idx_target)}_limit_submit_failed",
                                        )
                                        if not ok_fb:
                                            placed_all_tps = False
                                    else:
                                        placed_all_tps = False
                    elif stage_idx_target is not None and not desired_tps:
                        placed_all_tps = False

                    # Revalidar si la etapa objetivo esta viva tras submit.
                    try:
                        _orders = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                        existing_tp = _extract_tp_orders(symbol_orders, "SHORT", tp_mode_detect)
                        st_new = get_tp_state(symbol, "SHORT")

                        if stage_idx_target is None:
                            placed_all_tps = True
                        else:
                            stage_live = False
                            oid_target = _stage_order_id_from_state(st_new, stage_idx_target)
                            if oid_target and "orderId" in existing_tp.columns:
                                ids = existing_tp["orderId"].astype(str).str.strip().tolist()
                                stage_live = oid_target in ids
                            if not stage_live and stage_price_ref is not None:
                                existing_tp_prices = _extract_tp_price_set(existing_tp, symbol)
                                stage_live = any(abs((stage_price_ref - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                            next_idx_after = _next_tp_idx_from_stage(st_new.get("tp_stage", "none"))
                            placed_all_tps = stage_live or (next_idx_after != stage_idx_target)
                    except Exception:
                        pass
                else:
                    target_tp_count = len(split_qtys)
                    for idx, tp_px in enumerate(desired_tps[:target_tp_count]):
                        tp_idx = idx + 1
                        tp_px = _sanitize_trigger_price(float(tp_px), symbol, "SHORT", "TAKE_PROFIT_MARKET", market_ref)
                        exists = any(abs((tp_px - ex_px) / ex_px) < 1e-4 for ex_px in existing_tp_prices) if existing_tp_prices else False
                        if exists:
                            continue
                        tp_qty = split_qtys[idx]
                        if tp_qty <= 0:
                            continue
                        ok, tp_details = _post_with_retry(
                            symbol,
                            tp_qty,
                            0,
                            tp_px,
                            "SHORT",
                            "TAKE_PROFIT_MARKET",
                            "BUY",
                            return_details=True,
                        )
                        if ok:
                            time.sleep(0.3)
                            existing_tp_prices.add(tp_px)
                            set_tp_submitted(
                                symbol,
                                "SHORT",
                                tp_idx=tp_idx,
                                order_id=tp_details.get("order_id", ""),
                                qty=tp_qty,
                                price=tp_px,
                                submit_position_qty=pos_qty,
                                tp_mode=tp_mode_effective,
                                fill_confirmation_mode=tp_fill_mode,
                            )
                            emit_lifecycle_event(
                                f"tp{tp_idx}_submitted",
                                "INFO",
                                symbol=str(symbol).upper(),
                                position_side="SHORT",
                                qty=tp_qty,
                                tp_price=tp_px,
                                order_id=tp_details.get("order_id", ""),
                                source="colocando_TK_SL",
                            )
                            append_execution_ledger_event(
                                f"tp{tp_idx}_submitted",
                                data_quality="actual",
                                source="colocando_TK_SL_submit",
                                request_id=tp_details.get("request_id", ""),
                                order_id=tp_details.get("order_id", ""),
                                symbol=str(symbol).upper(),
                                side="BUY",
                                position_side="SHORT",
                                order_type="TAKE_PROFIT_MARKET",
                                submitted_price=0,
                                stop_price=tp_px,
                                submit_qty=tp_qty,
                                submit_time_utc=tp_details.get("submit_time_utc", ""),
                                partial_fill_status="unknown",
                            )
                        else:
                            placed_all_tps = False

                if exito_sl and placed_all_tps:
                    df_posiciones.drop(index, inplace=True)

                # Fallbacks adicionales: garantizar al menos una protección.
                # Refrescar snapshot para evitar duplicar envíos con estado stale.
                try:
                    _ = obteniendo_ordenes_pendientes()
                except Exception:
                    pass
                df_ordenes = _load_orders_register_df()
                symbol_orders = df_ordenes[df_ordenes['symbol'] == symbol]
                existing_tp = _extract_tp_orders(symbol_orders, "SHORT", tp_mode_detect)
                existing_sl = symbol_orders[symbol_orders['type'] == 'STOP_MARKET']
                if existing_sl.empty:
                    sl_px = _sanitize_trigger_price(float(sl_level), symbol, "SHORT", "STOP_MARKET", market_ref)
                    ok_sl = _post_with_retry(symbol, position_qty, 0, sl_px, "SHORT", "STOP_MARKET", "BUY")
                    time.sleep(0.3)
                    if ok_sl:
                        try:
                            _ = obteniendo_ordenes_pendientes()
                            df_tmp = pd.read_csv('./archivos/order_id_register.csv')
                            df_tmp = _normalize_orders_df(df_tmp)
                            m = df_tmp[(df_tmp['symbol'] == symbol) & (df_tmp['type'] == 'STOP_MARKET')].copy()
                            order_id = None
                            if not m.empty and 'stopPrice' in m.columns:
                                m['stopPrice'] = pd.to_numeric(m['stopPrice'], errors='coerce')
                                m['diff'] = (m['stopPrice'] - float(sl_px)).abs() / max(abs(float(sl_px)), 1e-9)
                                m = m.sort_values('diff')
                                if not m.empty and m['diff'].iloc[0] < 1e-3:
                                    order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                            _append_sl_watch(symbol, float(sl_px), 'SHORT', order_id)
                        except Exception:
                            _append_sl_watch(symbol, float(sl_px), 'SHORT', None)
                if tp_mode_effective != "partial_limit_tp" and existing_tp.empty and desired_tps:
                    tp1_px = _sanitize_trigger_price(float(desired_tps[0]), symbol, "SHORT", "TAKE_PROFIT_MARKET", market_ref)
                    try:
                        pos_qty = abs(float(positionAmt))
                    except Exception:
                        pos_qty = 0.0
                    tp_qty = _round_step(pos_qty, step_sz)
                    if tp_qty > 0:
                        ok, tp_details = _post_with_retry(
                            symbol,
                            tp_qty,
                            0,
                            tp1_px,
                            "SHORT",
                            "TAKE_PROFIT_MARKET",
                            "BUY",
                            return_details=True,
                        )
                        if ok:
                            set_tp_submitted(
                                symbol,
                                "SHORT",
                                tp_idx=1,
                                order_id=tp_details.get("order_id", ""),
                                qty=tp_qty,
                                price=tp1_px,
                                tp_mode=tp_mode_effective,
                                fill_confirmation_mode=tp_fill_mode,
                            )
                        time.sleep(0.3)

        except Exception as e:
            print(f"Error al configurar SL/TP para {symbol}: {e}")
            pass

    # Guardando Posiciones
    df_posiciones.to_csv('./archivos/position_id_register.csv', index=False)


#Cerrando Posiciones antiguas
def filtrando_posiciones_antiguas() -> pd.DataFrame:
    try:
        # Cargar los datos
        data = pd.read_csv('./archivos/order_id_register.csv')
        
        # Ajustar por zona horaria sumando 5 horas al tiempo actual
        # Tiempo Server AWS
        current_time = pd.Timestamp.now() - timedelta(hours=9)
        # Tiempo Mac
        # current_time = pd.Timestamp.now() + timedelta(hours=5)
        
        # Comprobar si la columna 'symbol' está en el DataFrame
        if 'symbol' not in data.columns:
            raise KeyError("La columna 'symbol' no se encuentra en el DataFrame.")
        
        # Filtro de columnas
        data_filtered = data[['symbol', 'orderId', 'type', 'time', 'stopPrice']].copy()
        data_filtered['time'] = pd.to_datetime(data_filtered['time'], unit='ms')
        
        # Calcular la diferencia de tiempo
        data_filtered['time_difference'] = (current_time - data_filtered['time']).dt.total_seconds() / 60
        
        # Filtrar entradas con más de 1 minuto de diferencia y de tipo 'STOP_MARKET'
        data_filtered = data_filtered[(data_filtered['time_difference'] > 1) & (data_filtered['type'] == 'STOP_MARKET')]
        
        # Remover duplicados basado en 'symbol'
        data_filtered = data_filtered.drop_duplicates(subset='symbol')
        
        # Resetear el índice
        data_filtered.reset_index(drop=True, inplace=True)
        
        return data_filtered

    except FileNotFoundError:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío

    except KeyError as e:
        return pd.DataFrame(columns=['symbol', 'orderId', 'type', 'time', 'stopPrice'])  # Retorna un DataFrame con las columnas esperadas pero vacío
    


def unrealized_profit_positions():
    # Cargar los indicadores
    df_indicadores = pd.read_csv('./archivos/indicadores.csv', low_memory=False)
    # Limpiar nombres de columnas y símbolo
    df_indicadores.columns = df_indicadores.columns.str.strip()
    df_indicadores['symbol'] = df_indicadores['symbol'].str.strip().str.upper()
    
    # Verificar que las columnas necesarias existen
    required_columns = ['symbol', 'close', 'Stop_Loss_Long', 'Stop_Loss_Short']
    missing_columns = [col for col in required_columns if col not in df_indicadores.columns]
    if missing_columns:
        print(f"Las siguientes columnas faltan en 'indicadores.csv': {missing_columns}")
        return
    
    # Convertir símbolos a mayúsculas para asegurar coincidencia
    df_indicadores['symbol'] = df_indicadores['symbol'].str.upper()
    
    # Agrupar por 'symbol' y obtener la última fila de cada grupo
    latest_values = df_indicadores.groupby('symbol').last().reset_index()

    # Cargar parametros por símbolo (para be_trigger) desde pkg/best_prod.json
    params_by_symbol = {}
    try:
        _best_path = str(BEST_PROD_PATH)
        if os.path.exists(_best_path):
            with open(_best_path, 'r') as _f:
                _prod = json.load(_f) or []
            params_by_symbol = {str(x.get('symbol', '')).upper(): (x.get('params') or {}) for x in _prod if isinstance(x, dict)}
    except Exception as _e:
        params_by_symbol = {}

    # Obtener datos filtrados de la función anterior
    data_filtered = filtrando_posiciones_antiguas()
    
    # Verificar si 'data_filtered' no está vacío
    if data_filtered.empty:
        # Silenciado: print("No hay posiciones antiguas para procesar.")
        return
    
    # Convertir símbolos a mayúsculas
    data_filtered['symbol'] = data_filtered['symbol'].str.upper()
    
    # Extraer la lista de símbolos
    symbols = data_filtered['symbol'].tolist()
    
    for symbol in symbols:
        # Silenciado: print(f"Procesando símbolo: {symbol}")

        # Obtener datos del símbolo en 'latest_values'
        symbol_data = latest_values[latest_values['symbol'] == symbol]

        # Verificar si 'symbol_data' está vacío
        if symbol_data.empty:
            # Silenciado: print(f"No hay datos de indicadores para el símbolo: {symbol}")
            continue

        # Obtener el precio actual
        precio_actual = symbol_data['close'].iloc[0]

        # Obtener datos de posición utilizando 'total_positions'
        result = total_positions(symbol)

        # Verificar si 'result' es None o no tiene suficientes datos
        if not result or result[0] is None:
            # Silenciado: print(f"No hay datos de posición para el símbolo: {symbol}")
            continue  # Saltar a la siguiente iteración del bucle

        # Desempaquetar el resultado
        symbol_result, positionSide, price, positionAmt, unrealizedProfit = result

        # Parámetro de break-even por símbolo (default 0.0 desactivado)
        p = params_by_symbol.get(str(symbol).upper(), {})
        try:
            be_trigger = float(p.get('be_trigger', 0.0))
        except Exception:
            be_trigger = 0.0
        TINY_BE = 0.0002  # 2 bps para cubrir fees/ticks
        tp_mode_runtime = get_tp_mode()
        be_after_tp1 = is_break_even_after_tp1_enabled()
        st_tp = get_tp_state(symbol, positionSide)
        tp_stage = str(st_tp.get("tp_stage", "none")).lower()
        tp1_confirmed = tp_stage in ("tp1_filled", "tp2_live", "tp2_filled", "tp3_live", "tp3_filled")
        allow_be_overlay = True
        if tp_mode_runtime == "partial_limit_tp" and be_after_tp1 and not tp1_confirmed:
            allow_be_overlay = False
            set_break_even_state(symbol, positionSide, "pending")
        elif tp_mode_runtime == "partial_limit_tp" and be_after_tp1 and tp1_confirmed:
            if str(st_tp.get("break_even_state", "inactive")).lower() != "active":
                set_break_even_state(symbol, positionSide, "pending")

        # Obtener el último valor de 'stopPrice' y 'orderId' para el símbolo
        filtered_data = data_filtered[data_filtered['symbol'] == symbol]

        # Verificar si 'filtered_data' está vacío
        if filtered_data.empty:
            # Silenciado: print(f"No se encontraron datos de 'order_id_register.csv' para el símbolo: {symbol}")
            continue

        # Acceder de forma segura a 'stopPrice' y 'orderId'
        last_stop_price = filtered_data['stopPrice'].iloc[-1]
        orderId = filtered_data['orderId'].iloc[-1]

        # Asegurar que 'positionAmt' es numérico
        try:
            positionAmt = float(positionAmt)
        except ValueError:
            # Silenciado: print(f"Cantidad de posición inválida para {symbol}: {positionAmt}")
            continue
        position_qty = abs(positionAmt)

        if positionSide == 'LONG':
            stop_loss = symbol_data['Stop_Loss_Long'].iloc[0]
            potencial_nuevo_sl = stop_loss
            be_applied = False
            # Break-Even: si el avance supera be_trigger, subir SL a entrada + tiny
            if allow_be_overlay and be_trigger > 0.0:
                try:
                    be_price = float(price) * (1.0 + be_trigger)
                    if float(precio_actual) >= be_price:
                        be_stop = float(price) * (1.0 + TINY_BE)
                        potencial_nuevo_sl = max(potencial_nuevo_sl, be_stop)
                        be_applied = bool(potencial_nuevo_sl >= be_stop)
                except Exception:
                    pass
            if potencial_nuevo_sl > last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    set_sl_guard(symbol, positionSide, seconds=25)
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    _post_with_retry(symbol, position_qty, 0, potencial_nuevo_sl, "LONG", "STOP_MARKET", "SELL")
                    # Actualizar watch con el nuevo SL
                    try:
                        _ = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        df_ordenes = _normalize_orders_df(df_ordenes)
                        m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                        order_id = None
                        if not m.empty and 'stopPrice' in m.columns:
                            m['diff'] = (m['stopPrice'] - float(potencial_nuevo_sl)).abs() / float(potencial_nuevo_sl)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'LONG', order_id)
                    except Exception:
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'LONG', None)
                    if be_applied:
                        set_break_even_state(symbol, positionSide, "active")
                        emit_lifecycle_event(
                            "break_even_activated",
                            "INFO",
                            symbol=str(symbol).upper(),
                            position_side="LONG",
                            new_sl=float(potencial_nuevo_sl),
                            source="unrealized_profit_positions",
                        )
                        append_execution_ledger_event(
                            "break_even_activated",
                            data_quality="inferred",
                            source="unrealized_profit_positions",
                            symbol=str(symbol).upper(),
                            position_side="LONG",
                            order_type="STOP_MARKET",
                            stop_price=float(potencial_nuevo_sl),
                        )
                    print(f"Stop Loss actualizado para {symbol} (LONG) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (LONG) es suficientemente bueno, no se modifica.")

        elif positionSide == 'SHORT':
            stop_loss = symbol_data['Stop_Loss_Short'].iloc[0]
            potencial_nuevo_sl = stop_loss
            be_applied = False
            # Break-Even: si la ganancia supera be_trigger, bajar SL a entrada - tiny
            if allow_be_overlay and be_trigger > 0.0:
                try:
                    be_price = float(price) * (1.0 - be_trigger)
                    if float(precio_actual) <= be_price:
                        be_stop = float(price) * (1.0 - TINY_BE)
                        potencial_nuevo_sl = min(potencial_nuevo_sl, be_stop)
                        be_applied = bool(potencial_nuevo_sl <= be_stop)
                except Exception:
                    pass
            if potencial_nuevo_sl < last_stop_price and potencial_nuevo_sl != last_stop_price:
                try:
                    set_sl_guard(symbol, positionSide, seconds=25)
                    pkg.bingx.cancel_order(symbol, orderId)
                    time.sleep(1)
                    _post_with_retry(symbol, position_qty, 0, potencial_nuevo_sl, "SHORT", "STOP_MARKET", "BUY")
                    # Actualizar watch con el nuevo SL
                    try:
                        _ = obteniendo_ordenes_pendientes()
                        df_ordenes = pd.read_csv('./archivos/order_id_register.csv')
                        df_ordenes = _normalize_orders_df(df_ordenes)
                        m = df_ordenes[(df_ordenes['symbol']==symbol) & (df_ordenes['type']=='STOP_MARKET')].copy()
                        order_id = None
                        if not m.empty and 'stopPrice' in m.columns:
                            m['diff'] = (m['stopPrice'] - float(potencial_nuevo_sl)).abs() / float(potencial_nuevo_sl)
                            m = m.sort_values('diff')
                            if not m.empty and m['diff'].iloc[0] < 1e-3:
                                order_id = m['orderId'].iloc[0] if 'orderId' in m.columns else None
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'SHORT', order_id)
                    except Exception:
                        _append_sl_watch(symbol, float(potencial_nuevo_sl), 'SHORT', None)
                    if be_applied:
                        set_break_even_state(symbol, positionSide, "active")
                        emit_lifecycle_event(
                            "break_even_activated",
                            "INFO",
                            symbol=str(symbol).upper(),
                            position_side="SHORT",
                            new_sl=float(potencial_nuevo_sl),
                            source="unrealized_profit_positions",
                        )
                        append_execution_ledger_event(
                            "break_even_activated",
                            data_quality="inferred",
                            source="unrealized_profit_positions",
                            symbol=str(symbol).upper(),
                            position_side="SHORT",
                            order_type="STOP_MARKET",
                            stop_price=float(potencial_nuevo_sl),
                        )

                    print(f"Stop Loss actualizado para {symbol} (SHORT) a {potencial_nuevo_sl}")
                except Exception as e:
                    print(f"Error al actualizar el Stop Loss para {symbol}: {e}")
            else:
                print(f"SL actual para {symbol} (SHORT) es suficientemente bueno, no se modifica.")
        else:
            # Silenciado: print(f"positionSide desconocido para {symbol}: {positionSide}")
            continue
  
