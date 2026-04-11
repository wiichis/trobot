"""Utilidades de alertas Telegram para monitoreo (separado de logica de estrategia).

Diseno:
- Config-driven (enabled, categorias, throttle, dedupe).
- Reusa credenciales existentes en pkg/credentials.py sin importar el paquete pkg.
- Estado local para cooldown/dedupe en archivo JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)
    if not isinstance(data, dict):
        return dict(default)
    out = dict(default)
    out.update(data)
    return out


def _load_monitor_cfg(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _env(*names: str) -> str:
    for name in names:
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""


def _read_credentials_from_env_or_aws() -> Tuple[Optional[str], Optional[str]]:
    token = _env("TROBOT_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
    chat_id = _env("TROBOT_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")
    if token and chat_id:
        return token, chat_id

    secret_id = _env("TROBOT_AWS_SECRET_ID")
    region_name = _env("TROBOT_AWS_REGION", "AWS_DEFAULT_REGION") or "us-east-1"
    if not secret_id:
        return (token or None), (chat_id or None)

    try:
        import boto3  # type: ignore
        client = boto3.client("secretsmanager", region_name=region_name)
        resp = client.get_secret_value(SecretId=secret_id)
        secret_string = resp.get("SecretString", "")
        data = json.loads(secret_string) if secret_string else {}
        if isinstance(data, dict):
            token = token or str(data.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
            chat_id = chat_id or str(data.get("TELEGRAM_CHAT_ID", "") or "").strip()
    except Exception:
        pass
    return (token or None), (chat_id or None)


def _read_credentials_from_file(repo_root: Path) -> Tuple[Optional[str], Optional[str]]:
    token, chat_id = _read_credentials_from_env_or_aws()
    if token and chat_id:
        return token, chat_id

    cred_path = repo_root / "pkg" / "credentials.py"
    if not cred_path.exists():
        return token, chat_id
    ns: Dict[str, Any] = {}
    try:
        code = cred_path.read_text(encoding="utf-8")
        exec(compile(code, str(cred_path), "exec"), ns, ns)
    except Exception:
        return token, chat_id
    file_token = ns.get("token")
    file_chat_id = ns.get("chatID") or ns.get("chat_id")
    token = token or (str(file_token).strip() if file_token else None)
    chat_id = chat_id or (str(file_chat_id).strip() if file_chat_id else None)
    return token, chat_id


def _msg_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deep_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _default_telegram_cfg() -> Dict[str, Any]:
    return {
        "enabled": False,
        "parse_mode": "",
        "chat_id_override": "",
        "send_timeout_sec": 12,
        "strategy_profile_name": "bingx_candidate",
        "categories_enabled": {
            "bot_started": True,
            "bot_stopped": True,
            "entry_order_submitted": True,
            "entry_order_filled": True,
            "trade_signal_detected": True,
            "take_profit_hit": True,
            "stop_loss_hit": True,
            "entry_order_canceled_or_expired": True,
            "tp1_submitted": True,
            "tp1_filled": True,
            "tp2_submitted": True,
            "tp2_filled": True,
            "tp3_submitted": True,
            "tp3_filled": True,
            "tp1_failed": True,
            "tp2_failed": True,
            "tp3_failed": True,
            "legacy_fallback_used": True,
            "break_even_activated": True,
            "tp_order_canceled_or_replaced": True,
            "monitoring_run_completed": True,
            "monitoring_run_failed": True,
            "portfolio_summary": True,
            "trade_closed_summary": True,
            "concentration_warning": True,
            "symbol_dependency_warning": True,
            "pnl_below_expectation": True,
            "cost_ratio_warning": True,
            "execution_quality_warning": True,
            "edge_degrading": True,
            "trade_count_too_low": True,
            "trade_frequency_abnormal": True,
            "symbol_mix_shift": True,
        },
        "throttle": {
            "default_cooldown_sec": 900,
            "dedupe_enabled": True,
            "by_category": {
                "bot_started": 60,
                "bot_stopped": 60,
                "entry_order_submitted": 20,
                "entry_order_filled": 20,
                "trade_signal_detected": 20,
                "take_profit_hit": 45,
                "stop_loss_hit": 45,
                "entry_order_canceled_or_expired": 90,
                "tp1_submitted": 20,
                "tp1_filled": 30,
                "tp2_submitted": 20,
                "tp2_filled": 30,
                "tp3_submitted": 20,
                "tp3_filled": 30,
                "tp1_failed": 45,
                "tp2_failed": 45,
                "tp3_failed": 45,
                "legacy_fallback_used": 90,
                "break_even_activated": 45,
                "tp_order_canceled_or_replaced": 90,
                "monitoring_run_completed": 300,
                "portfolio_summary": 3600,
                "trade_closed_summary": 900,
                "concentration_warning": 1800,
                "monitoring_run_failed": 60,
            },
        },
        "state_file": "archivos/backtesting/paper_live_monitor/telegram_alert_state.json",
    }


@dataclass
class TelegramSendResult:
    sent: bool
    detail: str


class TelegramAlerter:
    def __init__(self, *, config_path: Path, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.config_path = Path(config_path)
        full_cfg = _load_monitor_cfg(self.config_path)

        cfg_raw = _deep_get(full_cfg, "telegram", default={}) or {}
        self._explicit_enabled = isinstance(cfg_raw, dict) and ("enabled" in cfg_raw)
        cfg = _default_telegram_cfg()
        if isinstance(cfg_raw, dict):
            for k, v in cfg_raw.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    merged = dict(cfg.get(k) or {})
                    merged.update(v)
                    cfg[k] = merged
                else:
                    cfg[k] = v
        self.cfg = cfg

        token, chat_id = _read_credentials_from_file(self.repo_root)
        self.token = token
        override_chat = str(self.cfg.get("chat_id_override", "") or "").strip()
        self.chat_id = override_chat if override_chat else chat_id

        state_file = str(self.cfg.get("state_file", "")).strip()
        self.state_path = (self.repo_root / state_file) if state_file else (self.repo_root / "archivos/backtesting/paper_live_monitor/telegram_alert_state.json")

    @property
    def enabled(self) -> bool:
        if self._explicit_enabled:
            return bool(self.cfg.get("enabled", False))
        # Si no existe configuracion explicita, habilitar automaticamente cuando
        # hay secretos validos disponibles.
        return bool(self.token and self.chat_id)

    @property
    def profile_name(self) -> str:
        return str(self.cfg.get("strategy_profile_name", "bingx_candidate"))

    def _load_state(self) -> Dict[str, Any]:
        default = {"last_sent_at": {}, "last_sent_hash": {}}
        return _load_json(self.state_path, default=default)

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")

    def _category_enabled(self, category: str) -> bool:
        cats = self.cfg.get("categories_enabled", {})
        if not isinstance(cats, dict):
            return True
        val = cats.get(category)
        return True if val is None else bool(val)

    def _cooldown_seconds(self, category: str) -> int:
        throttle = self.cfg.get("throttle", {})
        default_cd = int(throttle.get("default_cooldown_sec", 900))
        by_cat = throttle.get("by_category", {})
        if isinstance(by_cat, dict) and category in by_cat:
            try:
                return int(by_cat.get(category))
            except Exception:
                return default_cd
        return default_cd

    def _dedupe_enabled(self) -> bool:
        throttle = self.cfg.get("throttle", {})
        return bool(throttle.get("dedupe_enabled", True))

    def _can_send(self, *, category: str, text_hash: str, now: datetime, force: bool) -> Tuple[bool, str]:
        if force:
            return True, "force"
        state = self._load_state()
        last_sent_at = _deep_get(state, "last_sent_at", category, default="")
        last_hash = _deep_get(state, "last_sent_hash", category, default="")

        if self._dedupe_enabled() and last_hash and last_hash == text_hash:
            return False, "deduped_same_message"

        prev = _parse_iso(str(last_sent_at))
        if prev is None:
            return True, "ok_first_send"
        cd = self._cooldown_seconds(category)
        elapsed = (now - prev).total_seconds()
        if elapsed < cd:
            return False, f"throttled_{int(cd - elapsed)}s"
        return True, "ok_cooldown_pass"

    _SEVERITY_EMOJI = {
        "INFO": "ℹ️",
        "WARN": "⚠️",
        "CRITICAL": "🚨",
        "ERROR": "🚨",
    }

    _CATEGORY_LABEL = {
        "bot_started": "Bot iniciado",
        "bot_stopped": "Bot detenido",
        "trade_signal_detected": "Señal detectada",
        "entry_order_submitted": "Orden de entrada enviada",
        "entry_order_filled": "Orden de entrada ejecutada",
        "entry_order_canceled_or_expired": "Orden cancelada/expirada",
        "take_profit_hit": "Take Profit alcanzado",
        "stop_loss_hit": "Stop Loss alcanzado",
        "tp1_submitted": "TP1 enviado",
        "tp1_filled": "TP1 ejecutado",
        "tp2_submitted": "TP2 enviado",
        "tp2_filled": "TP2 ejecutado",
        "tp3_submitted": "TP3 enviado",
        "tp3_filled": "TP3 ejecutado",
        "tp1_failed": "TP1 falló",
        "tp2_failed": "TP2 falló",
        "tp3_failed": "TP3 falló",
        "break_even_activated": "Break-even activado",
        "legacy_fallback_used": "Fallback legacy usado",
        "runtime_storage_warning": "Problema de almacenamiento",
        "execution_quality_warning": "Alerta de ejecución",
        "concentration_warning": "Alerta de concentración",
        "monitoring_run_completed": "Monitoreo completado",
        "monitoring_run_failed": "Monitoreo fallido",
    }

    def _build_header(self, *, severity: str, category: str, ts_utc: str) -> str:
        sev = str(severity).upper()
        emoji = self._SEVERITY_EMOJI.get(sev, "📌")
        label = self._CATEGORY_LABEL.get(category, category.replace("_", " ").title())
        return f"{emoji} *{label}*"

    def send(self, *, category: str, severity: str, body: str, ts_utc: str, force: bool = False) -> TelegramSendResult:
        if not self.enabled:
            return TelegramSendResult(sent=False, detail="telegram_disabled")
        if not self._category_enabled(category):
            return TelegramSendResult(sent=False, detail=f"category_disabled:{category}")
        if not self.token or not self.chat_id:
            return TelegramSendResult(sent=False, detail="missing_token_or_chat_id")

        text = self._build_header(severity=severity, category=category, ts_utc=ts_utc)
        if body:
            text = text + "\n" + body
        if len(text) > 3900:
            text = text[:3900] + "..."

        now = _now_utc()
        h = _msg_hash(text)
        ok_to_send, reason = self._can_send(category=category, text_hash=h, now=now, force=force)
        if not ok_to_send:
            return TelegramSendResult(sent=False, detail=reason)

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload: Dict[str, Any] = {"chat_id": self.chat_id, "text": text}
        parse_mode = str(self.cfg.get("parse_mode", "") or "").strip()
        if parse_mode:
            payload["parse_mode"] = parse_mode
        timeout = int(self.cfg.get("send_timeout_sec", 12))
        try:
            resp = requests.post(url, data=payload, timeout=timeout)
        except Exception as exc:
            return TelegramSendResult(sent=False, detail=f"telegram_request_error:{exc}")
        if not resp.ok:
            detail = f"telegram_status={resp.status_code}"
            try:
                body = (resp.text or "")[:220]
                detail += f":{body}"
            except Exception:
                pass
            return TelegramSendResult(sent=False, detail=detail)

        state = self._load_state()
        if not isinstance(state.get("last_sent_at"), dict):
            state["last_sent_at"] = {}
        if not isinstance(state.get("last_sent_hash"), dict):
            state["last_sent_hash"] = {}
        state["last_sent_at"][category] = _to_iso(now)
        state["last_sent_hash"][category] = h
        self._save_state(state)
        return TelegramSendResult(sent=True, detail="ok")
