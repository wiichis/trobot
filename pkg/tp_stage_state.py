"""Estado runtime de etapas TP/BE para orquestacion segura (Patch 5A).

No ejecuta ordenes. Solo mantiene estado persistente por (symbol, position_side)
para coordinar proteccion TP/SL y break-even.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
TP_STAGE_STATE_CSV = REPO_ROOT / "archivos" / "tp_stage_state.csv"

TP_STAGE_VALUES = (
    "none",
    "tp1_live",
    "tp1_filled",
    "tp2_live",
    "tp2_filled",
    "tp3_live",
    "tp3_filled",
)
BREAK_EVEN_VALUES = ("inactive", "pending", "active")

TP_STAGE_COLUMNS = [
    "symbol",
    "position_side",
    "tp_mode",
    "tp_stage",
    "break_even_state",
    "tp_fill_confirmation_mode",
    "tp1_order_id",
    "tp2_order_id",
    "tp3_order_id",
    "tp1_qty",
    "tp2_qty",
    "tp3_qty",
    "tp1_price",
    "tp2_price",
    "tp3_price",
    "tp1_submit_position_qty",
    "tp2_submit_position_qty",
    "tp3_submit_position_qty",
    "sl_guard_until_utc",
    "updated_at_utc",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso_utc() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _norm_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _norm_side(position_side: object) -> str:
    side = str(position_side or "").strip().upper()
    if side not in ("LONG", "SHORT"):
        return ""
    return side


def _safe_float_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    txt = str(value).strip()
    if txt == "" or txt.lower() in ("none", "nan", "na"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_state_df() -> pd.DataFrame:
    if not TP_STAGE_STATE_CSV.exists():
        return pd.DataFrame(columns=TP_STAGE_COLUMNS)
    try:
        df = pd.read_csv(TP_STAGE_STATE_CSV)
    except Exception:
        return pd.DataFrame(columns=TP_STAGE_COLUMNS)
    for c in TP_STAGE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["position_side"] = df["position_side"].astype(str).str.upper().str.strip()
    return df[TP_STAGE_COLUMNS].copy()


def _save_state_df(df: pd.DataFrame) -> None:
    TP_STAGE_STATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in TP_STAGE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    out = out[TP_STAGE_COLUMNS]
    out.to_csv(TP_STAGE_STATE_CSV, index=False)


def _default_row(symbol: str, position_side: str) -> Dict:
    return {
        "symbol": symbol,
        "position_side": position_side,
        "tp_mode": "legacy_market_tp",
        "tp_stage": "none",
        "break_even_state": "inactive",
        "tp_fill_confirmation_mode": "inferred",
        "tp1_order_id": "",
        "tp2_order_id": "",
        "tp3_order_id": "",
        "tp1_qty": None,
        "tp2_qty": None,
        "tp3_qty": None,
        "tp1_price": None,
        "tp2_price": None,
        "tp3_price": None,
        "tp1_submit_position_qty": None,
        "tp2_submit_position_qty": None,
        "tp3_submit_position_qty": None,
        "sl_guard_until_utc": "",
        "updated_at_utc": _now_iso_utc(),
    }


def get_tp_state(symbol: object, position_side: object) -> Dict:
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return _default_row(sym, side)
    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if not m.any():
        return _default_row(sym, side)
    row = df[m].tail(1).iloc[0].to_dict()
    row["symbol"] = sym
    row["position_side"] = side
    return row


def upsert_tp_state(
    symbol: object,
    position_side: object,
    *,
    tp_mode: Optional[str] = None,
    tp_stage: Optional[str] = None,
    break_even_state: Optional[str] = None,
    tp_fill_confirmation_mode: Optional[str] = None,
    tp1_order_id: Optional[str] = None,
    tp2_order_id: Optional[str] = None,
    tp3_order_id: Optional[str] = None,
    tp1_qty=None,
    tp2_qty=None,
    tp3_qty=None,
    tp1_price=None,
    tp2_price=None,
    tp3_price=None,
    tp1_submit_position_qty=None,
    tp2_submit_position_qty=None,
    tp3_submit_position_qty=None,
    sl_guard_until_utc: Optional[str] = None,
) -> Dict:
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return _default_row(sym, side)

    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if m.any():
        row = df[m].tail(1).iloc[0].to_dict()
        df = df[~m].copy()
    else:
        row = _default_row(sym, side)

    row["symbol"] = sym
    row["position_side"] = side
    if tp_mode is not None:
        row["tp_mode"] = str(tp_mode).strip().lower()
    if tp_stage is not None:
        tp_stage_v = str(tp_stage).strip().lower()
        if tp_stage_v in TP_STAGE_VALUES:
            row["tp_stage"] = tp_stage_v
    if break_even_state is not None:
        be_v = str(break_even_state).strip().lower()
        if be_v in BREAK_EVEN_VALUES:
            row["break_even_state"] = be_v
    if tp_fill_confirmation_mode is not None:
        row["tp_fill_confirmation_mode"] = str(tp_fill_confirmation_mode).strip().lower()

    if tp1_order_id is not None:
        row["tp1_order_id"] = str(tp1_order_id).strip()
    if tp2_order_id is not None:
        row["tp2_order_id"] = str(tp2_order_id).strip()
    if tp3_order_id is not None:
        row["tp3_order_id"] = str(tp3_order_id).strip()

    if tp1_qty is not None:
        row["tp1_qty"] = _safe_float_or_none(tp1_qty)
    if tp2_qty is not None:
        row["tp2_qty"] = _safe_float_or_none(tp2_qty)
    if tp3_qty is not None:
        row["tp3_qty"] = _safe_float_or_none(tp3_qty)
    if tp1_price is not None:
        row["tp1_price"] = _safe_float_or_none(tp1_price)
    if tp2_price is not None:
        row["tp2_price"] = _safe_float_or_none(tp2_price)
    if tp3_price is not None:
        row["tp3_price"] = _safe_float_or_none(tp3_price)
    if tp1_submit_position_qty is not None:
        row["tp1_submit_position_qty"] = _safe_float_or_none(tp1_submit_position_qty)
    if tp2_submit_position_qty is not None:
        row["tp2_submit_position_qty"] = _safe_float_or_none(tp2_submit_position_qty)
    if tp3_submit_position_qty is not None:
        row["tp3_submit_position_qty"] = _safe_float_or_none(tp3_submit_position_qty)

    if sl_guard_until_utc is not None:
        row["sl_guard_until_utc"] = str(sl_guard_until_utc).strip()

    row["updated_at_utc"] = _now_iso_utc()
    df.loc[len(df)] = {c: row.get(c, "") for c in TP_STAGE_COLUMNS}
    _save_state_df(df)
    return row


def clear_tp_state(symbol: object, position_side: object) -> None:
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return
    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if not m.any():
        return
    df = df[~m].copy()
    _save_state_df(df)


def set_tp_submitted(
    symbol: object,
    position_side: object,
    tp_idx: int,
    *,
    order_id: str = "",
    qty=None,
    price=None,
    submit_position_qty=None,
    tp_mode: Optional[str] = None,
    fill_confirmation_mode: Optional[str] = None,
) -> Dict:
    idx = int(tp_idx)
    stage = "tp1_live" if idx <= 1 else ("tp2_live" if idx == 2 else "tp3_live")
    kwargs = {
        "tp_stage": stage,
        "tp_mode": tp_mode,
        "tp_fill_confirmation_mode": fill_confirmation_mode,
    }
    if idx == 1:
        kwargs.update(
            {
                "tp1_order_id": order_id,
                "tp1_qty": qty,
                "tp1_price": price,
                "tp1_submit_position_qty": submit_position_qty,
            }
        )
    elif idx == 2:
        kwargs.update(
            {
                "tp2_order_id": order_id,
                "tp2_qty": qty,
                "tp2_price": price,
                "tp2_submit_position_qty": submit_position_qty,
            }
        )
    else:
        kwargs.update(
            {
                "tp3_order_id": order_id,
                "tp3_qty": qty,
                "tp3_price": price,
                "tp3_submit_position_qty": submit_position_qty,
            }
        )
    return upsert_tp_state(symbol, position_side, **kwargs)


def set_tp_filled(symbol: object, position_side: object, tp_idx: int) -> Dict:
    idx = int(tp_idx)
    if idx <= 1:
        return upsert_tp_state(symbol, position_side, tp_stage="tp1_filled", break_even_state="pending")
    if idx == 2:
        return upsert_tp_state(symbol, position_side, tp_stage="tp2_filled")
    return upsert_tp_state(symbol, position_side, tp_stage="tp3_filled")


def set_break_even_state(symbol: object, position_side: object, state: str) -> Dict:
    return upsert_tp_state(symbol, position_side, break_even_state=state)


def set_sl_guard(symbol: object, position_side: object, seconds: int = 20) -> Dict:
    try:
        sec = max(1, int(seconds))
    except Exception:
        sec = 20
    until = (_now_utc() + timedelta(seconds=sec)).isoformat().replace("+00:00", "Z")
    return upsert_tp_state(symbol, position_side, sl_guard_until_utc=until)


def is_sl_guard_active(symbol: object, position_side: object, now_utc: Optional[datetime] = None) -> bool:
    state = get_tp_state(symbol, position_side)
    raw = str(state.get("sl_guard_until_utc") or "").strip()
    if not raw:
        return False
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    now = now_utc if now_utc is not None else _now_utc()
    return bool(ts.to_pydatetime() > now)
