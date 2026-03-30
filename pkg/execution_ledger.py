"""Ledger append-only de ejecucion/fills para runtime paper/live.

Objetivo:
- Registrar eventos de ejecucion con esquema estable.
- Diferenciar claramente datos reales vs inferidos vs no disponibles.
- Servir como base para diagnosticos de calidad de ejecucion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXECUTION_LEDGER_PATH = REPO_ROOT / "archivos" / "execution_ledger.csv"

EXECUTION_LEDGER_COLUMNS = [
    "ts_utc",
    "event_type",
    "data_quality",
    "source",
    "request_id",
    "order_id",
    "symbol",
    "side",
    "position_side",
    "order_type",
    "intended_entry_price",
    "submitted_price",
    "actual_fill_price",
    "stop_price",
    "submit_qty",
    "fill_qty",
    "submit_time_utc",
    "fill_time_utc",
    "submit_to_fill_latency_sec",
    "maker_taker",
    "partial_fill_status",
    "cancel_reason",
    "close_reason",
    "close_qty",
    "raw_code",
    "raw_msg",
    "notes",
]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    if text == "" or text.lower() in ("none", "nan", "na"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_norm_text(value: Any, *, upper: bool = False) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if upper:
        text = text.upper()
    return text


def _compute_latency_sec(submit_time_utc: str, fill_time_utc: str) -> Optional[float]:
    submit_raw = str(submit_time_utc or "").strip()
    fill_raw = str(fill_time_utc or "").strip()
    if not submit_raw or not fill_raw:
        return None
    submit_ts = pd.to_datetime(submit_raw, utc=True, errors="coerce")
    fill_ts = pd.to_datetime(fill_raw, utc=True, errors="coerce")
    if pd.isna(submit_ts) or pd.isna(fill_ts):
        return None
    delta = (fill_ts - submit_ts).total_seconds()
    if delta < 0:
        return None
    return round(float(delta), 6)


def append_execution_ledger_event(
    event_type: str,
    *,
    data_quality: str = "inferred",
    source: str = "",
    ledger_path: Optional[Path] = None,
    ts_utc: Optional[str] = None,
    request_id: str = "",
    order_id: str = "",
    symbol: str = "",
    side: str = "",
    position_side: str = "",
    order_type: str = "",
    intended_entry_price: Any = None,
    submitted_price: Any = None,
    actual_fill_price: Any = None,
    stop_price: Any = None,
    submit_qty: Any = None,
    fill_qty: Any = None,
    submit_time_utc: str = "",
    fill_time_utc: str = "",
    submit_to_fill_latency_sec: Any = None,
    maker_taker: str = "",
    partial_fill_status: str = "",
    cancel_reason: str = "",
    close_reason: str = "",
    close_qty: Any = None,
    raw_code: Any = "",
    raw_msg: Any = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Agrega una fila al execution ledger.

    No lanza excepciones al caller; si falla, solo imprime error y retorna la fila.
    """

    row: Dict[str, Any] = {col: None for col in EXECUTION_LEDGER_COLUMNS}
    row["ts_utc"] = str(ts_utc or _now_iso_utc())
    row["event_type"] = _to_norm_text(event_type, upper=False)
    row["data_quality"] = _to_norm_text(data_quality, upper=False) or "inferred"
    row["source"] = _to_norm_text(source, upper=False)
    row["request_id"] = _to_norm_text(request_id)
    row["order_id"] = _to_norm_text(order_id)
    row["symbol"] = _to_norm_text(symbol, upper=True)
    row["side"] = _to_norm_text(side, upper=True)
    row["position_side"] = _to_norm_text(position_side, upper=True)
    row["order_type"] = _to_norm_text(order_type, upper=True)

    row["intended_entry_price"] = _to_optional_float(intended_entry_price)
    row["submitted_price"] = _to_optional_float(submitted_price)
    row["actual_fill_price"] = _to_optional_float(actual_fill_price)
    row["stop_price"] = _to_optional_float(stop_price)
    row["submit_qty"] = _to_optional_float(submit_qty)
    row["fill_qty"] = _to_optional_float(fill_qty)
    row["submit_time_utc"] = _to_norm_text(submit_time_utc)
    row["fill_time_utc"] = _to_norm_text(fill_time_utc)
    row["submit_to_fill_latency_sec"] = _to_optional_float(submit_to_fill_latency_sec)
    if row["submit_to_fill_latency_sec"] is None:
        row["submit_to_fill_latency_sec"] = _compute_latency_sec(
            row["submit_time_utc"],
            row["fill_time_utc"],
        )

    row["maker_taker"] = _to_norm_text(maker_taker, upper=False)
    row["partial_fill_status"] = _to_norm_text(partial_fill_status, upper=False)
    row["cancel_reason"] = _to_norm_text(cancel_reason, upper=False)
    row["close_reason"] = _to_norm_text(close_reason, upper=False)
    row["close_qty"] = _to_optional_float(close_qty)
    row["raw_code"] = _to_norm_text(raw_code)
    row["raw_msg"] = _to_norm_text(raw_msg)
    row["notes"] = _to_norm_text(notes)

    path = Path(ledger_path) if ledger_path is not None else DEFAULT_EXECUTION_LEDGER_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        pd.DataFrame([row], columns=EXECUTION_LEDGER_COLUMNS).to_csv(
            path,
            mode="a" if exists else "w",
            header=not exists,
            index=False,
        )
    except Exception as exc:
        print(f"Error escribiendo execution_ledger.csv: {exc}")
    return row

