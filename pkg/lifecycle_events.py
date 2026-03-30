"""Eventos de ciclo de vida + despacho Telegram unificado (runtime/monitoring).

Objetivo:
- Mantener observabilidad operacional en un CSV append-only.
- Reusar TelegramAlerter como canal principal de alertas.
- No acoplar esta capa a la logica de estrategia/ejecucion.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .telegram_alerts import TelegramAlerter


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TELEGRAM_CONFIG = REPO_ROOT / "archivos" / "backtesting" / "configs" / "paper_live_monitor_config.json"
LIFECYCLE_LOG_CSV = REPO_ROOT / "archivos" / "lifecycle_event_log.csv"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).strip()


def _build_body(fields: Dict[str, Any]) -> str:
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_fmt_value(value)}")
    body = " | ".join(parts)
    if len(body) > 1400:
        body = body[:1400] + "..."
    return body


def _append_event_row(row: Dict[str, Any]) -> None:
    cols = [
        "ts_utc",
        "category",
        "severity",
        "body",
        "fields_json",
        "telegram_sent",
        "telegram_detail",
    ]
    try:
        df_new = pd.DataFrame([row], columns=cols)
        exists = LIFECYCLE_LOG_CSV.exists() and LIFECYCLE_LOG_CSV.stat().st_size > 0
        LIFECYCLE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(LIFECYCLE_LOG_CSV, mode="a" if exists else "w", header=not exists, index=False)
    except Exception as exc:
        print(f"Error escribiendo lifecycle_event_log.csv: {exc}")


class LifecycleEventDispatcher:
    def __init__(self, config_path: Optional[Path] = None):
        cfg_path = Path(config_path) if config_path is not None else DEFAULT_TELEGRAM_CONFIG
        self.alerter = TelegramAlerter(
            config_path=cfg_path,
            repo_root=REPO_ROOT,
        )

    def emit(
        self,
        *,
        category: str,
        severity: str = "INFO",
        force: bool = False,
        **fields: Any,
    ) -> Dict[str, Any]:
        ts_utc = _now_iso_utc()
        body = _build_body(fields)
        result = self.alerter.send(
            category=str(category),
            severity=str(severity).upper(),
            body=body,
            ts_utc=ts_utc,
            force=bool(force),
        )
        row = {
            "ts_utc": ts_utc,
            "category": str(category),
            "severity": str(severity).upper(),
            "body": body,
            "fields_json": json.dumps(fields, ensure_ascii=True, default=str),
            "telegram_sent": bool(result.sent),
            "telegram_detail": str(result.detail),
        }
        _append_event_row(row)
        return {
            "sent": bool(result.sent),
            "detail": str(result.detail),
            "ts_utc": ts_utc,
        }


_DISPATCHER: Optional[LifecycleEventDispatcher] = None


def get_lifecycle_dispatcher() -> LifecycleEventDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = LifecycleEventDispatcher()
    return _DISPATCHER


def emit_lifecycle_event(
    category: str,
    severity: str = "INFO",
    *,
    force: bool = False,
    **fields: Any,
) -> Dict[str, Any]:
    try:
        dispatcher = get_lifecycle_dispatcher()
        return dispatcher.emit(category=category, severity=severity, force=force, **fields)
    except Exception as exc:
        ts_utc = _now_iso_utc()
        row = {
            "ts_utc": ts_utc,
            "category": str(category),
            "severity": str(severity).upper(),
            "body": _build_body(fields),
            "fields_json": json.dumps(fields, ensure_ascii=True, default=str),
            "telegram_sent": False,
            "telegram_detail": f"emit_error:{exc}",
        }
        _append_event_row(row)
        return {
            "sent": False,
            "detail": f"emit_error:{exc}",
            "ts_utc": ts_utc,
        }
