"""Configuracion runtime congelada para alinear live con benchmark validado.

Este modulo solo centraliza parametros operativos (sesion, lados, universo,
entry-style y cooldown). No cambia por si solo la logica de trading.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "archivos" / "backtesting" / "configs" / "live_benchmark_runtime.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "profile_name": "bingx_live_benchmark_v1",
    "reference": {
        "timeframe_combo": "30m_5m",
        "entry_style": "rsi_reversal",
        "session_profile": "liquid_utc_wo_13_18_20",
        "side_mode": "both",
        "universe_profile": "exclude_weak3",
        "long_filter_profile": "baseline",
    },
    "universe": {
        "mode": "from_best_prod",
        "symbols": [],
    },
    "session": {
        "entry_hours_utc": [6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 21, 22],
    },
    "side_mode": "both",
    "entry_style_overrides": {
        "logic": "strict",
        "require_rsi_cross": True,
        "fresh_breakout_only": False,
        "max_dist_emaslow": 0.008,
    },
    "long_filter_overrides": {},
    "timeframe": {
        "entry_tf": "5m",
        "htf_filter_enabled": True,
        "htf_tf": "30m",
        "htf_ema_fast": 50,
        "htf_ema_slow": 200,
        "htf_adx_min": 20.0,
        "htf_adx_period": 14,
    },
    "cooldown": {
        "mode": "fixed_minutes",
        "value": 50,  # benchmark cooldown=10 barras de 5m
    },
    "execution_entry": {
        "entry_mode": "limit_post_only",
        "entry_limit_offset_bps": 2.0,
        "entry_time_in_force": "PostOnly",
        "entry_post_only": True,
        "entry_market_fallback_on_error": True,
    },
    "execution_tp": {
        "tp_mode": "legacy_market_tp",
        "tp_partial_distribution": [0.33, 0.33, 0.34],
        "tp_limit_offset_bps": 0.0,
        "tp_reduce_only": True,
        "tp_one_at_a_time": True,
        "break_even_after_tp1": True,
        "tp_legacy_fallback_on_error": True,
        "tp_fill_confirmation_mode": "inferred",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _normalize_symbol(symbol: object) -> str:
    txt = str(symbol or "").strip().upper()
    return txt


def _normalize_hours(raw: object) -> Tuple[int, ...]:
    if raw is None:
        return tuple()
    if isinstance(raw, str):
        items = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]
    out = []
    for item in items:
        try:
            h = int(item)
        except Exception:
            continue
        if 0 <= h <= 23:
            out.append(h)
    return tuple(sorted(set(out)))


def _resolve_path(raw: object) -> Optional[Path]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    p = Path(txt).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _normalize_tp_distribution(raw: object) -> Tuple[float, float, float]:
    default = (0.33, 0.33, 0.34)
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return default
    vals = []
    for x in list(raw)[:3]:
        try:
            v = float(x)
        except Exception:
            return default
        vals.append(max(v, 0.0))
    s = sum(vals)
    if s <= 0:
        return default
    norm = [v / s for v in vals]
    return (float(norm[0]), float(norm[1]), float(norm[2]))


@lru_cache(maxsize=1)
def get_live_runtime_config() -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = DEFAULT_CONFIG_PATH
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cfg = _deep_merge(cfg, loaded)
        except Exception as exc:
            print(f"⚠️ Error leyendo runtime config {path}: {exc}")

    session = cfg.get("session") if isinstance(cfg.get("session"), dict) else {}
    session["entry_hours_utc"] = list(_normalize_hours(session.get("entry_hours_utc")))
    cfg["session"] = session

    side_mode = str(cfg.get("side_mode", "both")).strip().lower()
    if side_mode not in ("both", "short_only", "long_only"):
        side_mode = "both"
    cfg["side_mode"] = side_mode

    univ = cfg.get("universe") if isinstance(cfg.get("universe"), dict) else {}
    mode = str(univ.get("mode", "all")).strip().lower()
    if mode not in ("all", "exclude", "include"):
        mode = "all"
    symbols = univ.get("symbols") if isinstance(univ.get("symbols"), list) else []
    univ["mode"] = mode
    univ["symbols"] = sorted(set(_normalize_symbol(s) for s in symbols if str(s).strip()))
    cfg["universe"] = univ

    entry_cfg = cfg.get("execution_entry") if isinstance(cfg.get("execution_entry"), dict) else {}
    entry_mode = str(entry_cfg.get("entry_mode", "limit_post_only")).strip().lower()
    if entry_mode not in ("market", "limit_post_only"):
        entry_mode = "limit_post_only"
    try:
        entry_offset_bps = float(entry_cfg.get("entry_limit_offset_bps", 2.0))
    except Exception:
        entry_offset_bps = 2.0
    entry_cfg["entry_mode"] = entry_mode
    entry_cfg["entry_limit_offset_bps"] = max(0.0, entry_offset_bps)
    entry_cfg["entry_time_in_force"] = str(entry_cfg.get("entry_time_in_force", "PostOnly")).strip() or "PostOnly"
    entry_cfg["entry_post_only"] = bool(entry_cfg.get("entry_post_only", True))
    entry_cfg["entry_market_fallback_on_error"] = bool(entry_cfg.get("entry_market_fallback_on_error", True))
    cfg["execution_entry"] = entry_cfg

    tp_cfg = cfg.get("execution_tp") if isinstance(cfg.get("execution_tp"), dict) else {}
    tp_mode = str(tp_cfg.get("tp_mode", "legacy_market_tp")).strip().lower()
    if tp_mode not in ("legacy_market_tp", "partial_limit_tp"):
        tp_mode = "legacy_market_tp"
    tp_fill_mode = str(tp_cfg.get("tp_fill_confirmation_mode", "inferred")).strip().lower()
    if tp_fill_mode not in ("exchange_state", "inferred"):
        tp_fill_mode = "inferred"
    tp_cfg["tp_mode"] = tp_mode
    tp_cfg["tp_partial_distribution"] = list(_normalize_tp_distribution(tp_cfg.get("tp_partial_distribution")))
    try:
        tp_cfg["tp_limit_offset_bps"] = float(tp_cfg.get("tp_limit_offset_bps", 0.0))
    except Exception:
        tp_cfg["tp_limit_offset_bps"] = 0.0
    tp_cfg["tp_reduce_only"] = bool(tp_cfg.get("tp_reduce_only", True))
    tp_cfg["tp_one_at_a_time"] = bool(tp_cfg.get("tp_one_at_a_time", True))
    tp_cfg["break_even_after_tp1"] = bool(tp_cfg.get("break_even_after_tp1", True))
    tp_cfg["tp_legacy_fallback_on_error"] = bool(tp_cfg.get("tp_legacy_fallback_on_error", True))
    tp_cfg["tp_fill_confirmation_mode"] = tp_fill_mode
    cfg["execution_tp"] = tp_cfg

    return cfg


def reload_live_runtime_config() -> None:
    get_live_runtime_config.cache_clear()


def get_live_best_prod_path() -> Optional[Path]:
    cfg = get_live_runtime_config()
    return _resolve_path(cfg.get("best_prod_path"))


def get_allowed_entry_hours_utc() -> Tuple[int, ...]:
    cfg = get_live_runtime_config()
    session = cfg.get("session") if isinstance(cfg.get("session"), dict) else {}
    return _normalize_hours(session.get("entry_hours_utc"))


def is_entry_hour_allowed_utc(ts: Optional[datetime] = None) -> bool:
    allowed = get_allowed_entry_hours_utc()
    if not allowed:
        return True
    now = ts if ts is not None else datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return int(now.hour) in allowed


def get_side_mode_flags() -> Tuple[bool, bool]:
    side_mode = str(get_live_runtime_config().get("side_mode", "both")).lower()
    if side_mode == "short_only":
        return False, True
    if side_mode == "long_only":
        return True, False
    return True, True


def get_entry_style_overrides() -> Dict[str, Any]:
    cfg = get_live_runtime_config()
    out = cfg.get("entry_style_overrides")
    return dict(out) if isinstance(out, dict) else {}


def get_long_filter_overrides() -> Dict[str, Any]:
    cfg = get_live_runtime_config()
    out = cfg.get("long_filter_overrides")
    return dict(out) if isinstance(out, dict) else {}


def get_timeframe_overrides() -> Dict[str, Any]:
    cfg = get_live_runtime_config()
    out = cfg.get("timeframe")
    return dict(out) if isinstance(out, dict) else {}


def get_cooldown_minutes_override() -> Optional[int]:
    cfg = get_live_runtime_config()
    data = cfg.get("cooldown")
    if not isinstance(data, dict):
        return None
    mode = str(data.get("mode", "symbol_params")).strip().lower()
    if mode == "fixed_minutes":
        try:
            v = int(data.get("value"))
        except Exception:
            return None
        return v if v > 0 else None
    return None


def universe_allows_symbol(symbol: object) -> bool:
    sym = _normalize_symbol(symbol)
    if not sym:
        return False
    cfg = get_live_runtime_config()
    univ = cfg.get("universe") if isinstance(cfg.get("universe"), dict) else {}
    mode = str(univ.get("mode", "all")).lower()
    raw = univ.get("symbols") if isinstance(univ.get("symbols"), list) else []
    target = set(_normalize_symbol(x) for x in raw if str(x).strip())
    if mode == "exclude":
        return sym not in target
    if mode == "include":
        return sym in target
    return True


def get_tp_runtime_config() -> Dict[str, Any]:
    cfg = get_live_runtime_config()
    out = cfg.get("execution_tp")
    return dict(out) if isinstance(out, dict) else {}


def get_entry_runtime_config() -> Dict[str, Any]:
    cfg = get_live_runtime_config()
    out = cfg.get("execution_entry")
    return dict(out) if isinstance(out, dict) else {}


def get_entry_mode() -> str:
    mode = str(get_entry_runtime_config().get("entry_mode", "limit_post_only")).strip().lower()
    if mode not in ("market", "limit_post_only"):
        return "limit_post_only"
    return mode


def get_entry_limit_offset_bps() -> float:
    try:
        return max(0.0, float(get_entry_runtime_config().get("entry_limit_offset_bps", 2.0)))
    except Exception:
        return 2.0


def get_entry_time_in_force() -> str:
    tif = str(get_entry_runtime_config().get("entry_time_in_force", "PostOnly")).strip()
    return tif or "PostOnly"


def is_entry_post_only_enabled() -> bool:
    return bool(get_entry_runtime_config().get("entry_post_only", True))


def is_entry_market_fallback_on_error_enabled() -> bool:
    return bool(get_entry_runtime_config().get("entry_market_fallback_on_error", True))


def get_tp_mode() -> str:
    return str(get_tp_runtime_config().get("tp_mode", "legacy_market_tp")).strip().lower()


def get_tp_partial_distribution() -> Tuple[float, float, float]:
    raw = get_tp_runtime_config().get("tp_partial_distribution")
    return _normalize_tp_distribution(raw)


def get_tp_limit_offset_bps() -> float:
    try:
        return float(get_tp_runtime_config().get("tp_limit_offset_bps", 0.0))
    except Exception:
        return 0.0


def get_tp_reduce_only() -> bool:
    return bool(get_tp_runtime_config().get("tp_reduce_only", True))


def get_tp_one_at_a_time() -> bool:
    return bool(get_tp_runtime_config().get("tp_one_at_a_time", True))


def is_break_even_after_tp1_enabled() -> bool:
    return bool(get_tp_runtime_config().get("break_even_after_tp1", True))


def get_tp_legacy_fallback_on_error() -> bool:
    return bool(get_tp_runtime_config().get("tp_legacy_fallback_on_error", True))


def get_tp_fill_confirmation_mode() -> str:
    mode = str(get_tp_runtime_config().get("tp_fill_confirmation_mode", "inferred")).strip().lower()
    if mode not in ("exchange_state", "inferred"):
        return "inferred"
    return mode
