"""Monitoreo paper/live para estrategia candidata (sin tocar logica de estrategia).

Este modulo consume datos operativos existentes (PnL/equity/ordenes) y genera:
- resumen portfolio,
- resumen por simbolo,
- diagnosticos de concentracion,
- flags de drift,
- diagnosticos de calidad de ejecucion (con placeholders si faltan fills),
- alertas operativas legibles.

Diseno:
- append-friendly para corridas periodicas,
- umbrales centralizados en config,
- independiente de la logica de senales/ejecucion.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIG_PATH = Path("archivos/backtesting/configs/paper_live_monitor_config.json")
DEFAULT_OUT_DIR = Path("archivos/backtesting/paper_live_monitor")
DEFAULT_PNL_PATH = Path("archivos/PnL.csv")
DEFAULT_EQUITY_PATH = Path("archivos/ganancias.csv")
DEFAULT_ORDER_REGISTER_PATH = Path("archivos/order_id_register.csv")
DEFAULT_ORDER_SUBMIT_LOG_PATH = Path("archivos/order_submit_log.csv")
DEFAULT_ORDER_LIFECYCLE_LOG_PATH = Path("archivos/order_lifecycle_log.csv")
DEFAULT_EXECUTION_LEDGER_PATH = Path("archivos/execution_ledger.csv")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_timestamp_utc(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if ts.tzinfo is None:
        try:
            ts = ts.tz_localize("UTC")
        except Exception:
            return None
    else:
        try:
            ts = ts.tz_convert("UTC")
        except Exception:
            return None
    return ts


def _fmt_ts(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return ts.tz_convert("UTC").isoformat()
    except Exception:
        return ts.isoformat()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _default_config() -> Dict[str, Any]:
    return {
        "window": {
            "window_hours": 24,
            "recent_trades_n": 20,
        },
        "baseline": {
            "profile_summary_csv": "archivos/backtesting/exp90_robustness_report/robustness_profile_summary.csv",
            "symbol_breakdown_csv": "archivos/backtesting/exp90_robustness_report/robustness_symbol_breakdown.csv",
            "side_breakdown_csv": "archivos/backtesting/exp90_robustness_report/robustness_side_breakdown.csv",
            "concentration_csv": "archivos/backtesting/exp90_robustness_report/robustness_symbol_concentration.csv",
            "baseline_profile": "base",
            "baseline_window_days": 90,
            "fallback_profile": {
                "pnl_net": 1.43,
                "max_dd_pct": -0.0012,
                "winrate_pct": 80.95,
                "profit_factor": 4.101,
                "trades": 21,
                "cost_ratio": 0.3016,
            },
            "fallback_concentration": {
                "top1_share_pct": 69.9301,
                "top2_share_pct": 97.9021,
            },
            "fallback_symbol_mix_pct": {
                "HBAR-USDT": 69.9301,
                "DOGE-USDT": 27.9720,
                "XMR-USDT": 2.0979,
            },
            "fallback_side_pnl": {
                "long": 0.4830,
                "short": 0.9511,
            },
        },
        "execution_assumptions": {
            "expected_slippage_bps": 4.0,
            "expected_fee_bps_per_side": 5.0,
            "supports_fill_level_data": False,
        },
        "thresholds": {
            "concentration": {
                "top1_warn_pct": 70.0,
                "top2_warn_pct": 95.0,
                "top1_vs_baseline_delta_warn_pct": 5.0,
                "top2_vs_baseline_delta_warn_pct": 3.0,
            },
            "drift": {
                "min_trade_count_window": 8,
                "min_pnl_per_trade_ratio_vs_baseline": 0.50,
                "min_profit_factor_ratio_vs_baseline": 0.70,
                "max_cost_ratio_multiplier_vs_baseline": 1.35,
                "max_cost_ratio_absolute": 0.85,
                "trade_freq_low_ratio": 0.50,
                "trade_freq_high_ratio": 1.80,
                "symbol_mix_l1_warn": 0.45,
                "side_pnl_share_delta_warn": 0.35,
            },
            "alerts": {
                "symbol_dependency_top2_pct": 95.0,
                "trade_count_too_low": 6,
                "require_execution_fill_data": False,
                "min_order_acceptance_rate": 0.85,
                "min_pending_gone_ratio_proxy": 0.20,
                "min_pending_events_for_ratio": 5,
            },
        },
        "outputs": {
            "append": True,
        },
    }


def load_monitor_config(config_path: Optional[Path]) -> Dict[str, Any]:
    base = _default_config()
    if config_path is None:
        return base
    path = Path(config_path)
    if not path.exists():
        return base
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return base
    if not isinstance(loaded, dict):
        return base
    return _deep_merge(base, loaded)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _parse_bool_series(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin(["1", "true", "t", "yes", "y", "on"])


def _slice_by_ts(df: pd.DataFrame, ts_col: str, start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty or ts_col not in df.columns:
        return pd.DataFrame(columns=df.columns)
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out[out[ts_col].notna()]
    if start_ts is not None:
        out = out[out[ts_col] >= start_ts]
    if end_ts is not None:
        out = out[out[ts_col] <= end_ts]
    return out


def _load_income_table(pnl_csv: Path) -> pd.DataFrame:
    df = _read_csv_safe(pnl_csv)
    req = {"symbol", "incomeType", "income", "time"}
    if df.empty or not req.issubset(df.columns):
        return pd.DataFrame(columns=["symbol", "incomeType", "income", "time", "info", "tradeId"])
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["incomeType"] = out["incomeType"].astype(str).str.upper().str.strip()
    out["income"] = pd.to_numeric(out["income"], errors="coerce")
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if "info" not in out.columns:
        out["info"] = ""
    out["info"] = out["info"].astype(str)
    if "tradeId" not in out.columns:
        out["tradeId"] = ""
    out["tradeId"] = out["tradeId"].astype(str)
    out = out.dropna(subset=["symbol", "income", "time"]).copy()
    return out.sort_values("time").reset_index(drop=True)


def _load_equity_table(equity_csv: Path) -> pd.DataFrame:
    df = _read_csv_safe(equity_csv)
    if df.empty or "date" not in df.columns or "balance" not in df.columns:
        return pd.DataFrame(columns=["date", "balance"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out["balance"] = pd.to_numeric(out["balance"], errors="coerce")
    out = out.dropna(subset=["date", "balance"]).sort_values("date").reset_index(drop=True)
    return out


def _derive_side_from_realized_info(info: str) -> Optional[str]:
    txt = str(info).strip().lower()
    # En BingX income:
    # - "Sell to Close" => cierra LONG.
    # - "Buy to Close"  => cierra SHORT.
    if "sell to close" in txt:
        return "long"
    if "buy to close" in txt:
        return "short"
    return None


def _profit_factor(pnls: Iterable[float]) -> Optional[float]:
    vals = pd.Series(list(pnls), dtype=float)
    if vals.empty:
        return None
    pos = float(vals[vals > 0].sum())
    neg = float(-vals[vals < 0].sum())
    if neg <= 0:
        return None
    return pos / neg


def _slice_window(df: pd.DataFrame, window_hours: int) -> Tuple[pd.DataFrame, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df.empty:
        return df.copy(), None, None
    end_ts = _to_timestamp_utc(df["time"].max())
    if end_ts is None:
        return pd.DataFrame(columns=df.columns), None, None
    start_ts = end_ts - pd.Timedelta(hours=int(window_hours))
    sliced = df[df["time"] >= start_ts].copy()
    return sliced, start_ts, end_ts


def _max_drawdown_from_equity(eq: pd.DataFrame, start_ts: Optional[pd.Timestamp], end_ts: Optional[pd.Timestamp]) -> Optional[float]:
    if eq.empty:
        return None
    e = eq.copy()
    if start_ts is not None:
        e = e[e["date"] >= start_ts]
    if end_ts is not None:
        e = e[e["date"] <= end_ts]
    if e.empty:
        return None
    bal = e["balance"].astype(float).to_numpy()
    peak = np.maximum.accumulate(bal)
    valid = peak > 0
    if not valid.any():
        return None
    dd = np.full_like(bal, fill_value=np.nan, dtype=float)
    dd[valid] = (bal[valid] - peak[valid]) / peak[valid]
    try:
        return float(np.nanmin(dd))
    except Exception:
        return None


def _load_baseline_bundle(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bcfg = cfg.get("baseline", {}) if isinstance(cfg, dict) else {}
    profile_name = str(bcfg.get("baseline_profile", "base"))

    profile_df = _read_csv_safe(Path(str(bcfg.get("profile_summary_csv", ""))))
    symbol_df = _read_csv_safe(Path(str(bcfg.get("symbol_breakdown_csv", ""))))
    side_df = _read_csv_safe(Path(str(bcfg.get("side_breakdown_csv", ""))))
    conc_df = _read_csv_safe(Path(str(bcfg.get("concentration_csv", ""))))

    base_profile: Dict[str, Any]
    if not profile_df.empty and "profile" in profile_df.columns:
        pick = profile_df[profile_df["profile"] == profile_name]
        if pick.empty:
            pick = profile_df.iloc[[0]]
        base_profile = pick.iloc[0].to_dict()
    else:
        base_profile = dict(bcfg.get("fallback_profile", {}))

    symbol_mix: Dict[str, float] = {}
    if not symbol_df.empty and {"profile", "symbol", "pnl_net"}.issubset(symbol_df.columns):
        pick = symbol_df[symbol_df["profile"] == profile_name].copy()
        pick["symbol"] = pick["symbol"].astype(str).str.upper()
        pick["pnl_net"] = pd.to_numeric(pick["pnl_net"], errors="coerce").fillna(0.0)
        pos = pick[pick["pnl_net"] > 0].copy()
        total_pos = float(pos["pnl_net"].sum())
        if total_pos > 0:
            symbol_mix = {str(r["symbol"]): float(r["pnl_net"]) / total_pos for _, r in pos.iterrows()}
    if not symbol_mix:
        fb_mix = bcfg.get("fallback_symbol_mix_pct", {})
        if isinstance(fb_mix, dict):
            raw = {str(k).upper(): _safe_float(v) / 100.0 for k, v in fb_mix.items()}
            s = sum(max(v, 0.0) for v in raw.values())
            if s > 0:
                symbol_mix = {k: max(v, 0.0) / s for k, v in raw.items()}

    side_mix: Dict[str, float] = {}
    if not side_df.empty and {"profile", "side", "pnl_net"}.issubset(side_df.columns):
        pick = side_df[side_df["profile"] == profile_name].copy()
        pick["side"] = pick["side"].astype(str).str.lower().str.strip()
        pick["pnl_net"] = pd.to_numeric(pick["pnl_net"], errors="coerce").fillna(0.0)
        pos = pick[pick["pnl_net"] > 0].copy()
        total_pos = float(pos["pnl_net"].sum())
        if total_pos > 0:
            side_mix = {str(r["side"]): float(r["pnl_net"]) / total_pos for _, r in pos.iterrows()}
    if not side_mix:
        fb_side = bcfg.get("fallback_side_pnl", {})
        if isinstance(fb_side, dict):
            lval = max(_safe_float(fb_side.get("long", 0.0)), 0.0)
            sval = max(_safe_float(fb_side.get("short", 0.0)), 0.0)
            den = lval + sval
            if den > 0:
                side_mix = {"long": lval / den, "short": sval / den}

    concentration = {}
    if not conc_df.empty and {"profile", "top1_share_pct", "top2_share_pct"}.issubset(conc_df.columns):
        pick = conc_df[conc_df["profile"] == profile_name]
        if pick.empty:
            pick = conc_df.iloc[[0]]
        concentration = {
            "top1_share_pct": _safe_float(pick.iloc[0].get("top1_share_pct"), math.nan),
            "top2_share_pct": _safe_float(pick.iloc[0].get("top2_share_pct"), math.nan),
        }
    if not concentration:
        concentration = dict(bcfg.get("fallback_concentration", {}))

    baseline_window_days = _safe_float(bcfg.get("baseline_window_days"), 90.0)
    trades = _safe_float(base_profile.get("trades"), 0.0)
    trades_per_day = (trades / baseline_window_days) if baseline_window_days > 0 else 0.0
    pnl_net = _safe_float(base_profile.get("pnl_net"), 0.0)
    pnl_per_trade = (pnl_net / trades) if trades > 0 else 0.0

    return {
        "profile": base_profile,
        "symbol_mix": symbol_mix,
        "side_mix": side_mix,
        "concentration": concentration,
        "trades_per_day": trades_per_day,
        "pnl_per_trade": pnl_per_trade,
    }


def _portfolio_summary(
    df_window: pd.DataFrame,
    df_equity: pd.DataFrame,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    cfg: Dict[str, Any],
    run_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    realized = df_window[df_window["incomeType"] == "REALIZED_PNL"].copy()
    fees = df_window[df_window["incomeType"] == "TRADING_FEE"].copy()
    funding = df_window[df_window["incomeType"] == "FUNDING_FEE"].copy()

    realized["side"] = realized["info"].apply(_derive_side_from_realized_info)

    trade_count = int(len(realized))
    pnl_realized = float(realized["income"].sum()) if trade_count else 0.0
    fee_net = float(fees["income"].sum()) if not fees.empty else 0.0
    funding_net = float(funding["income"].sum()) if not funding.empty else 0.0
    pnl_net = pnl_realized + fee_net + funding_net

    fees_paid = float((-fees["income"].clip(upper=0.0)).sum()) if not fees.empty else 0.0
    funding_paid = float((-funding["income"].clip(upper=0.0)).sum()) if not funding.empty else 0.0

    wins = int((realized["income"] > 0).sum()) if trade_count else 0
    winrate = (wins / trade_count * 100.0) if trade_count else 0.0
    pf = _profit_factor(realized["income"].tolist()) if trade_count else None

    long_pnl = float(realized.loc[realized["side"] == "long", "income"].sum()) if trade_count else 0.0
    short_pnl = float(realized.loc[realized["side"] == "short", "income"].sum()) if trade_count else 0.0
    positive_side_pool = max(long_pnl, 0.0) + max(short_pnl, 0.0)
    long_share = (max(long_pnl, 0.0) / positive_side_pool * 100.0) if positive_side_pool > 0 else None
    short_share = (max(short_pnl, 0.0) / positive_side_pool * 100.0) if positive_side_pool > 0 else None

    realized_abs = abs(pnl_realized)
    cost_ratio = ((fees_paid + funding_paid) / realized_abs) if realized_abs > 0 else None

    max_dd_pct = _max_drawdown_from_equity(df_equity, start_ts, end_ts)
    expected_slippage_bps = _safe_float(cfg.get("execution_assumptions", {}).get("expected_slippage_bps"), math.nan)

    row = {
        "run_ts_utc": _fmt_ts(run_ts),
        "window_start_utc": _fmt_ts(start_ts),
        "window_end_utc": _fmt_ts(end_ts),
        "window_hours": _safe_int(cfg.get("window", {}).get("window_hours"), 24),
        "pnl_realized": round(pnl_realized, 6),
        "pnl_net": round(pnl_net, 6),
        "unrealized_pnl": None,
        "trade_count": trade_count,
        "winrate_pct": round(winrate, 4),
        "profit_factor": round(float(pf), 6) if pf is not None else None,
        "pnl_long": round(long_pnl, 6),
        "pnl_short": round(short_pnl, 6),
        "pnl_long_share_pct": round(float(long_share), 4) if long_share is not None else None,
        "pnl_short_share_pct": round(float(short_share), 4) if short_share is not None else None,
        "max_dd_pct": round(float(max_dd_pct), 6) if max_dd_pct is not None else None,
        "cost_ratio": round(float(cost_ratio), 6) if cost_ratio is not None else None,
        "fees_paid": round(fees_paid, 6),
        "fees_net": round(fee_net, 6),
        "funding_paid": round(funding_paid, 6),
        "funding_net": round(funding_net, 6),
        "slippage_expected_bps": round(expected_slippage_bps, 4) if not math.isnan(expected_slippage_bps) else None,
        "slippage_realized_proxy_bps": None,
        "notes": "unrealized/slippage_realized requieren fill-level logs",
    }
    return pd.DataFrame([row]), realized


def _symbol_summary(
    df_window: pd.DataFrame,
    realized: pd.DataFrame,
    portfolio_pnl_net: float,
    recent_trades_n: int,
    run_ts: pd.Timestamp,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if df_window.empty:
        return pd.DataFrame(
            columns=[
                "run_ts_utc",
                "window_start_utc",
                "window_end_utc",
                "symbol",
                "trades",
                "pnl_realized",
                "pnl_net",
                "winrate_pct",
                "profit_factor",
                "pnl_share_pct",
                "pnl_long",
                "pnl_short",
                "long_share_pct",
                "short_share_pct",
                "recent_trades",
                "recent_pnl_realized",
                "recent_pnl_share_pct",
            ]
        )

    rr = realized.sort_values("time").copy()
    if not rr.empty:
        rr["side"] = rr["info"].apply(_derive_side_from_realized_info)
    recent_rr = rr.tail(max(0, int(recent_trades_n))).copy()

    recent_pool = float(recent_rr.loc[recent_rr["income"] > 0, "income"].sum()) if not recent_rr.empty else 0.0
    # Shares por simbolo sobre pool positivo (evita porcentajes inestables cuando el total es <= 0).
    symbol_pnl_tmp: Dict[str, float] = {}
    for sym, grp in df_window.groupby("symbol"):
        g = grp.copy()
        g_realized = g[g["incomeType"] == "REALIZED_PNL"].copy()
        fees_net = float(g.loc[g["incomeType"] == "TRADING_FEE", "income"].sum())
        funding_net = float(g.loc[g["incomeType"] == "FUNDING_FEE", "income"].sum())
        pnl_realized = float(g_realized["income"].sum()) if not g_realized.empty else 0.0
        symbol_pnl_tmp[str(sym).upper()] = pnl_realized + fees_net + funding_net
    positive_pool = float(sum(max(v, 0.0) for v in symbol_pnl_tmp.values()))

    out_rows: List[Dict[str, Any]] = []

    for sym, grp in df_window.groupby("symbol"):
        g = grp.copy()
        g_realized = g[g["incomeType"] == "REALIZED_PNL"].copy()
        if not g_realized.empty:
            g_realized["side"] = g_realized["info"].apply(_derive_side_from_realized_info)

        fees_net = float(g.loc[g["incomeType"] == "TRADING_FEE", "income"].sum())
        funding_net = float(g.loc[g["incomeType"] == "FUNDING_FEE", "income"].sum())
        pnl_realized = float(g_realized["income"].sum()) if not g_realized.empty else 0.0
        pnl_net = pnl_realized + fees_net + funding_net
        trades = int(len(g_realized))
        wins = int((g_realized["income"] > 0).sum()) if trades else 0
        winrate = (wins / trades * 100.0) if trades else 0.0
        pf = _profit_factor(g_realized["income"].tolist()) if trades else None

        long_pnl = float(g_realized.loc[g_realized["side"] == "long", "income"].sum()) if trades else 0.0
        short_pnl = float(g_realized.loc[g_realized["side"] == "short", "income"].sum()) if trades else 0.0
        side_pos_pool = max(long_pnl, 0.0) + max(short_pnl, 0.0)
        long_share = (max(long_pnl, 0.0) / side_pos_pool * 100.0) if side_pos_pool > 0 else None
        short_share = (max(short_pnl, 0.0) / side_pos_pool * 100.0) if side_pos_pool > 0 else None

        share = (max(pnl_net, 0.0) / positive_pool * 100.0) if positive_pool > 0 else None

        sym_recent = recent_rr[recent_rr["symbol"] == sym]
        recent_trades = int(len(sym_recent))
        recent_pnl_realized = float(sym_recent["income"].sum()) if recent_trades else 0.0
        recent_share = (max(recent_pnl_realized, 0.0) / recent_pool * 100.0) if recent_pool > 0 else None

        out_rows.append(
            {
                "run_ts_utc": _fmt_ts(run_ts),
                "window_start_utc": _fmt_ts(start_ts),
                "window_end_utc": _fmt_ts(end_ts),
                "symbol": str(sym).upper(),
                "trades": trades,
                "pnl_realized": round(pnl_realized, 6),
                "pnl_net": round(pnl_net, 6),
                "winrate_pct": round(winrate, 4),
                "profit_factor": round(float(pf), 6) if pf is not None else None,
                "pnl_share_pct": round(float(share), 4) if share is not None else None,
                "pnl_long": round(long_pnl, 6),
                "pnl_short": round(short_pnl, 6),
                "long_share_pct": round(float(long_share), 4) if long_share is not None else None,
                "short_share_pct": round(float(short_share), 4) if short_share is not None else None,
                "recent_trades": recent_trades,
                "recent_pnl_realized": round(recent_pnl_realized, 6),
                "recent_pnl_share_pct": round(float(recent_share), 4) if recent_share is not None else None,
            }
        )

    df_out = pd.DataFrame(out_rows)
    if df_out.empty:
        return df_out
    return df_out.sort_values(["pnl_net", "trades"], ascending=[False, False]).reset_index(drop=True)


def _concentration_summary(
    symbol_df: pd.DataFrame,
    cfg: Dict[str, Any],
    run_ts: pd.Timestamp,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
) -> pd.DataFrame:
    cols = [
        "run_ts_utc",
        "window_start_utc",
        "window_end_utc",
        "top1_symbol",
        "top1_share",
        "top2_symbols",
        "top2_share",
        "positive_pnl_pool",
        "recent_top1_symbol",
        "recent_top1_share",
        "recent_top2_symbols",
        "recent_top2_share",
        "top1_threshold",
        "top2_threshold",
        "concentration_warning",
    ]
    if symbol_df.empty:
        return pd.DataFrame(columns=cols)

    th = cfg.get("thresholds", {}).get("concentration", {})
    top1_th = _safe_float(th.get("top1_warn_pct"), 70.0)
    top2_th = _safe_float(th.get("top2_warn_pct"), 95.0)

    d = symbol_df.copy()
    d["pnl_net"] = pd.to_numeric(d["pnl_net"], errors="coerce").fillna(0.0)
    d["recent_pnl_realized"] = pd.to_numeric(d["recent_pnl_realized"], errors="coerce").fillna(0.0)

    cur = d[d["pnl_net"] > 0].sort_values("pnl_net", ascending=False).reset_index(drop=True)
    cur_pool = float(cur["pnl_net"].sum())
    cur_top1_symbol = str(cur.iloc[0]["symbol"]) if len(cur) >= 1 else None
    cur_top1 = float(cur.iloc[0]["pnl_net"]) if len(cur) >= 1 else 0.0
    cur_top2 = float(cur.iloc[:2]["pnl_net"].sum()) if len(cur) >= 2 else cur_top1
    cur_top2_symbols = ",".join(cur.iloc[:2]["symbol"].astype(str).tolist()) if len(cur) >= 2 else cur_top1_symbol
    cur_top1_share = (cur_top1 / cur_pool * 100.0) if cur_pool > 0 else None
    cur_top2_share = (cur_top2 / cur_pool * 100.0) if cur_pool > 0 else None

    rec = d[d["recent_pnl_realized"] > 0].sort_values("recent_pnl_realized", ascending=False).reset_index(drop=True)
    rec_pool = float(rec["recent_pnl_realized"].sum())
    rec_top1_symbol = str(rec.iloc[0]["symbol"]) if len(rec) >= 1 else None
    rec_top1 = float(rec.iloc[0]["recent_pnl_realized"]) if len(rec) >= 1 else 0.0
    rec_top2 = float(rec.iloc[:2]["recent_pnl_realized"].sum()) if len(rec) >= 2 else rec_top1
    rec_top2_symbols = ",".join(rec.iloc[:2]["symbol"].astype(str).tolist()) if len(rec) >= 2 else rec_top1_symbol
    rec_top1_share = (rec_top1 / rec_pool * 100.0) if rec_pool > 0 else None
    rec_top2_share = (rec_top2 / rec_pool * 100.0) if rec_pool > 0 else None

    concentration_warning = False
    if cur_top1_share is not None and cur_top1_share >= top1_th:
        concentration_warning = True
    if cur_top2_share is not None and cur_top2_share >= top2_th:
        concentration_warning = True

    row = {
        "run_ts_utc": _fmt_ts(run_ts),
        "window_start_utc": _fmt_ts(start_ts),
        "window_end_utc": _fmt_ts(end_ts),
        "top1_symbol": cur_top1_symbol,
        "top1_share": round(float(cur_top1_share), 4) if cur_top1_share is not None else None,
        "top2_symbols": cur_top2_symbols,
        "top2_share": round(float(cur_top2_share), 4) if cur_top2_share is not None else None,
        "positive_pnl_pool": round(cur_pool, 6),
        "recent_top1_symbol": rec_top1_symbol,
        "recent_top1_share": round(float(rec_top1_share), 4) if rec_top1_share is not None else None,
        "recent_top2_symbols": rec_top2_symbols,
        "recent_top2_share": round(float(rec_top2_share), 4) if rec_top2_share is not None else None,
        "top1_threshold": top1_th,
        "top2_threshold": top2_th,
        "concentration_warning": bool(concentration_warning),
    }
    return pd.DataFrame([row], columns=cols)


def _l1_symbol_mix_distance(current: Dict[str, float], baseline: Dict[str, float]) -> float:
    keys = sorted(set(current.keys()) | set(baseline.keys()))
    if not keys:
        return 0.0
    return float(sum(abs(_safe_float(current.get(k, 0.0)) - _safe_float(baseline.get(k, 0.0))) for k in keys))


def _current_symbol_mix(symbol_df: pd.DataFrame) -> Dict[str, float]:
    if symbol_df.empty:
        return {}
    d = symbol_df.copy()
    d["pnl_net"] = pd.to_numeric(d["pnl_net"], errors="coerce").fillna(0.0)
    pos = d[d["pnl_net"] > 0].copy()
    total = float(pos["pnl_net"].sum())
    if total <= 0:
        return {}
    return {str(r["symbol"]).upper(): float(r["pnl_net"]) / total for _, r in pos.iterrows()}


def _current_side_mix(portfolio_row: pd.Series) -> Dict[str, float]:
    l = _safe_float(portfolio_row.get("pnl_long"), 0.0)
    s = _safe_float(portfolio_row.get("pnl_short"), 0.0)
    lp = max(l, 0.0)
    sp = max(s, 0.0)
    den = lp + sp
    if den <= 0:
        return {}
    return {"long": lp / den, "short": sp / den}


def _drift_flags(
    portfolio_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    baseline: Dict[str, Any],
    cfg: Dict[str, Any],
    run_ts: pd.Timestamp,
) -> pd.DataFrame:
    cols = ["run_ts_utc", "flag", "status", "current_value", "baseline_value", "threshold", "details"]
    if portfolio_df.empty:
        return pd.DataFrame(columns=cols)

    p = portfolio_df.iloc[0]
    base = baseline.get("profile", {})
    base_pptr = _safe_float(baseline.get("pnl_per_trade"), 0.0)
    base_pf = _safe_float(base.get("profit_factor"), 0.0)
    base_cost = _safe_float(base.get("cost_ratio"), 0.0)
    base_trades_day = _safe_float(baseline.get("trades_per_day"), 0.0)

    th = cfg.get("thresholds", {}).get("drift", {})
    min_trades = _safe_int(th.get("min_trade_count_window"), 8)
    min_pptr_ratio = _safe_float(th.get("min_pnl_per_trade_ratio_vs_baseline"), 0.5)
    min_pf_ratio = _safe_float(th.get("min_profit_factor_ratio_vs_baseline"), 0.7)
    max_cost_mult = _safe_float(th.get("max_cost_ratio_multiplier_vs_baseline"), 1.35)
    max_cost_abs = _safe_float(th.get("max_cost_ratio_absolute"), 0.85)
    low_freq_ratio = _safe_float(th.get("trade_freq_low_ratio"), 0.5)
    high_freq_ratio = _safe_float(th.get("trade_freq_high_ratio"), 1.8)
    symbol_mix_l1_warn = _safe_float(th.get("symbol_mix_l1_warn"), 0.45)
    side_share_delta_warn = _safe_float(th.get("side_pnl_share_delta_warn"), 0.35)

    trade_count = _safe_int(p.get("trade_count"), 0)
    pnl_net = _safe_float(p.get("pnl_net"), 0.0)
    cur_pptr = (pnl_net / trade_count) if trade_count > 0 else 0.0
    cur_pf = _safe_float(p.get("profit_factor"), 0.0)
    cur_cost = _safe_float(p.get("cost_ratio"), math.nan)
    window_hours = max(_safe_float(p.get("window_hours"), 24.0), 1.0)
    cur_trades_day = trade_count * (24.0 / window_hours)

    edge_degrading = False
    edge_details = []
    if trade_count >= min_trades:
        if base_pptr > 0 and cur_pptr < (base_pptr * min_pptr_ratio):
            edge_degrading = True
            edge_details.append("pnl_per_trade_bajo")
        if base_pf > 0 and cur_pf > 0 and cur_pf < (base_pf * min_pf_ratio):
            edge_degrading = True
            edge_details.append("profit_factor_bajo")
    else:
        edge_details.append("muestra_baja_para_edge")

    cost_drift = False
    if not (isinstance(cur_cost, float) and math.isnan(cur_cost)):
        if base_cost > 0 and cur_cost > (base_cost * max_cost_mult):
            cost_drift = True
        if cur_cost > max_cost_abs:
            cost_drift = True

    trade_freq_abnormal = False
    if base_trades_day > 0:
        lo = base_trades_day * low_freq_ratio
        hi = base_trades_day * high_freq_ratio
        if cur_trades_day < lo or cur_trades_day > hi:
            trade_freq_abnormal = True
    else:
        lo = hi = None

    conc = concentration_df.iloc[0].to_dict() if not concentration_df.empty else {}
    cur_top1 = _safe_float(conc.get("top1_share"), math.nan)
    cur_top2 = _safe_float(conc.get("top2_share"), math.nan)
    base_top1 = _safe_float(baseline.get("concentration", {}).get("top1_share_pct"), math.nan)
    base_top2 = _safe_float(baseline.get("concentration", {}).get("top2_share_pct"), math.nan)
    cth = cfg.get("thresholds", {}).get("concentration", {})
    delta1_warn = _safe_float(cth.get("top1_vs_baseline_delta_warn_pct"), 5.0)
    delta2_warn = _safe_float(cth.get("top2_vs_baseline_delta_warn_pct"), 3.0)
    concentration_rising = False
    if not math.isnan(cur_top1) and not math.isnan(base_top1) and cur_top1 > (base_top1 + delta1_warn):
        concentration_rising = True
    if not math.isnan(cur_top2) and not math.isnan(base_top2) and cur_top2 > (base_top2 + delta2_warn):
        concentration_rising = True

    cur_mix = _current_symbol_mix(symbol_df)
    base_mix = baseline.get("symbol_mix", {})
    if trade_count >= min_trades and cur_mix:
        mix_dist = _l1_symbol_mix_distance(cur_mix, base_mix)
        symbol_mix_shift = mix_dist >= symbol_mix_l1_warn
        mix_details = "l1_distance_on_positive_pnl_mix"
    else:
        mix_dist = 0.0
        symbol_mix_shift = False
        mix_details = "insufficient_sample"

    cur_side_mix = _current_side_mix(p)
    base_side_mix = baseline.get("side_mix", {})
    if trade_count >= min_trades and cur_side_mix:
        long_delta = abs(_safe_float(cur_side_mix.get("long"), 0.0) - _safe_float(base_side_mix.get("long"), 0.0))
        short_delta = abs(_safe_float(cur_side_mix.get("short"), 0.0) - _safe_float(base_side_mix.get("short"), 0.0))
        max_side_delta = max(long_delta, short_delta)
        long_short_balance_shift = max_side_delta >= side_share_delta_warn
        side_details = "max_abs_delta_on_side_positive_pnl_share"
    else:
        max_side_delta = 0.0
        long_short_balance_shift = False
        side_details = "insufficient_sample"

    rows: List[Dict[str, Any]] = [
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "edge_degrading",
            "status": bool(edge_degrading),
            "current_value": round(cur_pptr, 6),
            "baseline_value": round(base_pptr, 6),
            "threshold": round(base_pptr * min_pptr_ratio, 6) if base_pptr > 0 else None,
            "details": ";".join(edge_details) if edge_details else "",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "concentration_rising",
            "status": bool(concentration_rising),
            "current_value": round(cur_top2, 4) if not math.isnan(cur_top2) else None,
            "baseline_value": round(base_top2, 4) if not math.isnan(base_top2) else None,
            "threshold": round(base_top2 + delta2_warn, 4) if not math.isnan(base_top2) else None,
            "details": "top2_share_vs_baseline",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "trade_frequency_abnormal",
            "status": bool(trade_freq_abnormal),
            "current_value": round(cur_trades_day, 6),
            "baseline_value": round(base_trades_day, 6) if base_trades_day > 0 else None,
            "threshold": f"[{round(lo, 4) if lo is not None else None},{round(hi, 4) if hi is not None else None}]",
            "details": "trades_per_day_window",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "cost_drift",
            "status": bool(cost_drift),
            "current_value": round(cur_cost, 6) if not (isinstance(cur_cost, float) and math.isnan(cur_cost)) else None,
            "baseline_value": round(base_cost, 6) if base_cost > 0 else None,
            "threshold": min(round(base_cost * max_cost_mult, 6), round(max_cost_abs, 6)) if base_cost > 0 else round(max_cost_abs, 6),
            "details": "cost_ratio_window",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "symbol_mix_shift",
            "status": bool(symbol_mix_shift),
            "current_value": round(mix_dist, 6),
            "baseline_value": 0.0,
            "threshold": symbol_mix_l1_warn,
            "details": mix_details,
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "flag": "long_short_balance_shift",
            "status": bool(long_short_balance_shift),
            "current_value": round(max_side_delta, 6),
            "baseline_value": 0.0,
            "threshold": side_share_delta_warn,
            "details": side_details,
        },
    ]
    return pd.DataFrame(rows, columns=cols)


def _execution_quality_summary(
    cfg: Dict[str, Any],
    run_ts: pd.Timestamp,
    start_ts: Optional[pd.Timestamp],
    end_ts: Optional[pd.Timestamp],
    order_register_csv: Path,
    order_submit_log_csv: Path,
    order_lifecycle_log_csv: Path,
    execution_ledger_csv: Path,
) -> pd.DataFrame:
    ex_cfg = cfg.get("execution_assumptions", {})
    supports_fill = bool(ex_cfg.get("supports_fill_level_data", False))
    open_orders = _read_csv_safe(order_register_csv)
    open_count = int(len(open_orders)) if not open_orders.empty else 0

    submit_raw = _read_csv_safe(order_submit_log_csv)
    submit = _slice_by_ts(submit_raw, "ts_utc", start_ts, end_ts)
    if not submit.empty:
        if "accepted" in submit.columns:
            submit["accepted_bool"] = _parse_bool_series(submit["accepted"])
        else:
            submit["accepted_bool"] = False
        if "order_type" in submit.columns:
            submit["order_type"] = submit["order_type"].astype(str).str.upper()
        else:
            submit["order_type"] = ""
    else:
        submit = pd.DataFrame(columns=["accepted_bool", "order_type", "price", "stop_price"])

    sub_total = int(len(submit))
    sub_acc = int(submit["accepted_bool"].sum()) if sub_total else 0
    sub_acc_rate = (sub_acc / sub_total) if sub_total else None

    entry_submit = submit[submit["order_type"] == "MARKET"].copy() if sub_total else pd.DataFrame()
    entry_total = int(len(entry_submit))
    entry_acc = int(entry_submit["accepted_bool"].sum()) if entry_total else 0
    entry_acc_rate = (entry_acc / entry_total) if entry_total else None

    lifecycle_raw = _read_csv_safe(order_lifecycle_log_csv)
    lifecycle = _slice_by_ts(lifecycle_raw, "ts_utc", start_ts, end_ts)
    if not lifecycle.empty:
        if "event_type" in lifecycle.columns:
            lifecycle["event_type"] = lifecycle["event_type"].astype(str).str.lower()
        else:
            lifecycle["event_type"] = ""
    pending_seen = int((lifecycle["event_type"] == "pending_seen").sum()) if not lifecycle.empty else 0
    pending_gone = int((lifecycle["event_type"] == "pending_gone").sum()) if not lifecycle.empty else 0
    pending_gone_ratio = (pending_gone / pending_seen) if pending_seen > 0 else None

    ledger_raw = _read_csv_safe(execution_ledger_csv)
    ledger = _slice_by_ts(ledger_raw, "ts_utc", start_ts, end_ts)
    if not ledger.empty:
        if "event_type" in ledger.columns:
            ledger["event_type"] = ledger["event_type"].astype(str).str.lower().str.strip()
        else:
            ledger["event_type"] = ""
        for col in [
            "intended_entry_price",
            "actual_fill_price",
            "submit_to_fill_latency_sec",
            "maker_taker",
            "partial_fill_status",
        ]:
            if col not in ledger.columns:
                ledger[col] = np.nan if col != "maker_taker" and col != "partial_fill_status" else ""
    else:
        ledger = pd.DataFrame(
            columns=[
                "event_type",
                "intended_entry_price",
                "actual_fill_price",
                "submit_to_fill_latency_sec",
                "maker_taker",
                "partial_fill_status",
            ]
        )

    ledger_events = int(len(ledger))
    entry_fill_events = int((ledger["event_type"] == "entry_order_filled").sum()) if ledger_events else 0
    intended_ser = pd.to_numeric(ledger["intended_entry_price"], errors="coerce") if ledger_events else pd.Series(dtype=float)
    actual_ser = pd.to_numeric(ledger["actual_fill_price"], errors="coerce") if ledger_events else pd.Series(dtype=float)
    latency_ser = pd.to_numeric(ledger["submit_to_fill_latency_sec"], errors="coerce") if ledger_events else pd.Series(dtype=float)

    actual_fill_available = bool(actual_ser.notna().any()) if not actual_ser.empty else False
    intended_available = bool(intended_ser.notna().any()) if not intended_ser.empty else False

    slippage_proxy_bps = None
    if not intended_ser.empty and not actual_ser.empty:
        valid = intended_ser.notna() & actual_ser.notna() & (intended_ser.abs() > 0)
        if bool(valid.any()):
            slip_bps = ((actual_ser[valid] - intended_ser[valid]) / intended_ser[valid]).abs() * 10000.0
            if not slip_bps.empty:
                slippage_proxy_bps = float(slip_bps.mean())

    latency_mean = None
    if not latency_ser.empty:
        latency_valid = latency_ser[latency_ser >= 0]
        if not latency_valid.empty:
            latency_mean = float(latency_valid.mean())

    maker_taker_available = False
    if ledger_events and "maker_taker" in ledger.columns:
        maker_taker_available = bool(
            ledger["maker_taker"].astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()
        )

    partial_fill_available = False
    if ledger_events and "partial_fill_status" in ledger.columns:
        pf = ledger["partial_fill_status"].astype(str).str.strip().str.lower()
        partial_fill_available = bool(pf[~pf.isin(["", "nan", "none", "unknown"])].any())

    intended_submit_available = False
    if sub_total:
        price_ok = pd.to_numeric(submit.get("price"), errors="coerce") if "price" in submit.columns else pd.Series(dtype=float)
        stop_ok = pd.to_numeric(submit.get("stop_price"), errors="coerce") if "stop_price" in submit.columns else pd.Series(dtype=float)
        intended_submit_available = bool(
            (price_ok.notna().any() if not price_ok.empty else False)
            or (stop_ok.notna().any() if not stop_ok.empty else False)
        )
    intended_available = bool(intended_available or intended_submit_available)

    fill_data_available = bool(supports_fill or actual_fill_available)

    details = (
        f"submit_total={sub_total};submit_accepted={sub_acc};submit_accept_rate={round(sub_acc_rate,4) if sub_acc_rate is not None else None};"
        f"entry_total={entry_total};entry_accept_rate={round(entry_acc_rate,4) if entry_acc_rate is not None else None};"
        f"pending_seen={pending_seen};pending_gone={pending_gone};pending_gone_ratio={round(pending_gone_ratio,4) if pending_gone_ratio is not None else None};"
        f"ledger_events={ledger_events};entry_fill_events={entry_fill_events};"
        f"latency_mean={round(latency_mean,4) if latency_mean is not None else None};"
        f"slippage_proxy_bps={round(slippage_proxy_bps,4) if slippage_proxy_bps is not None else None}"
    )

    row = {
        "run_ts_utc": _fmt_ts(run_ts),
        "window_start_utc": _fmt_ts(start_ts),
        "window_end_utc": _fmt_ts(end_ts),
        "fill_data_available": fill_data_available,
        "intended_price_available": intended_available,
        "actual_fill_price_available": actual_fill_available,
        "expected_slippage_bps": _safe_float(ex_cfg.get("expected_slippage_bps"), math.nan),
        "realized_slippage_proxy_bps": round(slippage_proxy_bps, 6) if slippage_proxy_bps is not None else None,
        "submit_to_fill_latency_sec_mean": round(latency_mean, 6) if latency_mean is not None else None,
        "execution_ledger_events": ledger_events,
        "entry_fill_events": entry_fill_events,
        "maker_taker_available": maker_taker_available,
        "partial_fill_status_available": partial_fill_available,
        "open_orders_count": open_count,
        "missed_fill_rate": round(1.0 - pending_gone_ratio, 6) if pending_gone_ratio is not None else None,
        "limit_fill_rate": round(pending_gone_ratio, 6) if pending_gone_ratio is not None else None,
        "details": details,
    }
    return pd.DataFrame([row])


def _alerts(
    portfolio_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    cfg: Dict[str, Any],
    run_ts: pd.Timestamp,
) -> pd.DataFrame:
    cols = ["run_ts_utc", "alert", "severity", "triggered", "value", "threshold", "message"]
    if portfolio_df.empty:
        return pd.DataFrame(columns=cols)

    p = portfolio_df.iloc[0]
    conc = concentration_df.iloc[0].to_dict() if not concentration_df.empty else {}
    drift_map = {str(r["flag"]): bool(r["status"]) for _, r in drift_df.iterrows()} if not drift_df.empty else {}
    ex = execution_df.iloc[0].to_dict() if not execution_df.empty else {}

    alert_cfg = cfg.get("thresholds", {}).get("alerts", {})
    top2_dep = _safe_float(alert_cfg.get("symbol_dependency_top2_pct"), 95.0)
    min_trades = _safe_int(alert_cfg.get("trade_count_too_low"), 6)
    require_fill_data = bool(alert_cfg.get("require_execution_fill_data", False))
    min_order_accept_rate = _safe_float(alert_cfg.get("min_order_acceptance_rate"), 0.85)
    min_pending_gone_ratio = _safe_float(alert_cfg.get("min_pending_gone_ratio_proxy"), 0.20)
    min_pending_events_for_ratio = _safe_int(alert_cfg.get("min_pending_events_for_ratio"), 5)

    top1_share = _safe_float(conc.get("top1_share"), math.nan)
    top2_share = _safe_float(conc.get("top2_share"), math.nan)
    trade_count = _safe_int(p.get("trade_count"), 0)
    cost_ratio = _safe_float(p.get("cost_ratio"), math.nan)
    exec_accept_rate = None
    pending_seen = 0
    pending_gone_ratio = None
    ex_details = str(ex.get("details", ""))
    try:
        parts = [x for x in ex_details.split(";") if "=" in x]
        kv = {k.strip(): v.strip() for k, v in (p.split("=", 1) for p in parts)}
        if kv.get("submit_accept_rate") not in (None, "", "None"):
            exec_accept_rate = float(kv.get("submit_accept_rate"))
        if kv.get("pending_seen") not in (None, "", "None"):
            pending_seen = int(float(kv.get("pending_seen")))
        if kv.get("pending_gone_ratio") not in (None, "", "None"):
            pending_gone_ratio = float(kv.get("pending_gone_ratio"))
    except Exception:
        exec_accept_rate = None
        pending_seen = 0
        pending_gone_ratio = None

    exec_warn = bool(require_fill_data and not bool(ex.get("fill_data_available", False)))
    if exec_accept_rate is not None and exec_accept_rate < min_order_accept_rate:
        exec_warn = True
    if pending_gone_ratio is not None and pending_seen >= min_pending_events_for_ratio and pending_gone_ratio < min_pending_gone_ratio:
        exec_warn = True

    rows = [
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "concentration_warning",
            "severity": "warning",
            "triggered": bool(conc.get("concentration_warning", False)),
            "value": round(top2_share, 4) if not math.isnan(top2_share) else None,
            "threshold": conc.get("top2_threshold"),
            "message": "Concentracion de ganancias elevada (top1/top2).",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "pnl_below_expectation",
            "severity": "warning",
            "triggered": bool(drift_map.get("edge_degrading", False)),
            "value": _safe_float(p.get("pnl_net"), 0.0),
            "threshold": "baseline_relative",
            "message": "El rendimiento reciente por trade/PF cae vs baseline.",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "cost_ratio_warning",
            "severity": "warning",
            "triggered": bool(drift_map.get("cost_drift", False)),
            "value": round(cost_ratio, 6) if not math.isnan(cost_ratio) else None,
            "threshold": "baseline_cost_ratio*mult",
            "message": "Deriva de costos detectada.",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "symbol_dependency_warning",
            "severity": "warning",
            "triggered": bool((not math.isnan(top2_share)) and top2_share >= top2_dep),
            "value": round(top2_share, 4) if not math.isnan(top2_share) else None,
            "threshold": top2_dep,
            "message": "Dependencia elevada en pocos simbolos (top2).",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "trade_count_too_low",
            "severity": "info",
            "triggered": bool(trade_count < min_trades),
            "value": trade_count,
            "threshold": min_trades,
            "message": "Muestra reciente baja para validar edge estadisticamente.",
        },
        {
            "run_ts_utc": _fmt_ts(run_ts),
            "alert": "execution_quality_warning",
            "severity": "info",
            "triggered": bool(exec_warn),
            "value": ex_details,
            "threshold": f"fill_data={require_fill_data}|min_accept={min_order_accept_rate}|min_pending_ratio={min_pending_gone_ratio}",
            "message": "Calidad de ejecucion degradada o sin data suficiente.",
        },
    ]
    return pd.DataFrame(rows, columns=cols)


def _human_summary(
    portfolio_df: pd.DataFrame,
    concentration_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
) -> Dict[str, str]:
    if portfolio_df.empty:
        return {
            "q1_expectations": "Sin datos en ventana.",
            "q2_concentration": "Sin datos en ventana.",
            "q3_hbar_doge_dependency": "Sin datos en ventana.",
            "q4_longs_value": "Sin datos en ventana.",
            "q5_costs_slippage": "Sin datos en ventana.",
            "q6_operational_decision": "review",
        }

    p = portfolio_df.iloc[0]
    drift_map = {str(r["flag"]): bool(r["status"]) for _, r in drift_df.iterrows()} if not drift_df.empty else {}
    conc = concentration_df.iloc[0].to_dict() if not concentration_df.empty else {}
    trade_count = _safe_int(p.get("trade_count"), 0)
    low_sample = trade_count < 6

    top2_symbols = str(conc.get("top2_symbols") or "")
    top2_share = _safe_float(conc.get("top2_share"), math.nan)
    top1_share = _safe_float(conc.get("top1_share"), math.nan)
    conc_warn = bool(conc.get("concentration_warning", False))
    pnl_long = _safe_float(p.get("pnl_long"), 0.0)
    pnl_short = _safe_float(p.get("pnl_short"), 0.0)

    if low_sample:
        q1 = "Muestra insuficiente en ventana para confirmar cercania al baseline."
    else:
        q1 = "Si, comportamiento cercano al baseline." if not drift_map.get("edge_degrading", False) else "No, hay drift de edge vs baseline."
    if conc_warn:
        q2 = "Concentracion alta en la ventana (top1/top2 por encima de umbral)."
    elif drift_map.get("concentration_rising", False):
        q2 = "Concentracion en aumento."
    else:
        q2 = "Concentracion estable o menor respecto al baseline."
    q3 = (
        f"Top2 actuales: {top2_symbols} ({top2_share:.2f}%)."
        if top2_symbols and not math.isnan(top2_share)
        else "Sin evidencia suficiente para dependencia HBAR/DOGE en esta ventana."
    )
    q4 = (
        f"Longs aportan ({pnl_long:.4f}) y shorts ({pnl_short:.4f})."
        if abs(pnl_long) > 0
        else f"Longs no aportan en la ventana; shorts explican la mayor parte ({pnl_short:.4f})."
    )
    q5 = (
        "Costos bajo control."
        if not drift_map.get("cost_drift", False)
        else "Costos empeorando vs baseline (cost drift)."
    )

    severe = bool(drift_map.get("edge_degrading", False))
    caution = bool(drift_map.get("concentration_rising", False)) or (
        (not math.isnan(top1_share) and top1_share >= 70.0)
        or (not math.isnan(top2_share) and top2_share >= 95.0)
    )
    if severe:
        decision = "review"
    elif low_sample:
        decision = "active_with_caution"
    elif caution:
        decision = "active_with_caution"
    else:
        decision = "active"

    return {
        "q1_expectations": q1,
        "q2_concentration": q2,
        "q3_hbar_doge_dependency": q3,
        "q4_longs_value": q4,
        "q5_costs_slippage": q5,
        "q6_operational_decision": decision,
    }


def _append_csv(df: pd.DataFrame, path: Path, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        return
    if append and path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


@dataclass
class MonitorResult:
    portfolio: pd.DataFrame
    symbol: pd.DataFrame
    concentration: pd.DataFrame
    drift_flags: pd.DataFrame
    execution_quality: pd.DataFrame
    alerts: pd.DataFrame
    summary: Dict[str, Any]
    output_files: Dict[str, str]


def run_monitor(
    *,
    pnl_csv: Path = DEFAULT_PNL_PATH,
    equity_csv: Path = DEFAULT_EQUITY_PATH,
    order_register_csv: Path = DEFAULT_ORDER_REGISTER_PATH,
    order_submit_log_csv: Path = DEFAULT_ORDER_SUBMIT_LOG_PATH,
    order_lifecycle_log_csv: Path = DEFAULT_ORDER_LIFECYCLE_LOG_PATH,
    execution_ledger_csv: Path = DEFAULT_EXECUTION_LEDGER_PATH,
    config_path: Optional[Path] = DEFAULT_CONFIG_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    override_window_hours: Optional[int] = None,
) -> MonitorResult:
    cfg = load_monitor_config(config_path)
    run_ts = pd.Timestamp.now(tz="UTC")

    if override_window_hours is not None:
        cfg["window"]["window_hours"] = int(override_window_hours)
    window_hours = max(1, _safe_int(cfg.get("window", {}).get("window_hours"), 24))
    recent_n = max(1, _safe_int(cfg.get("window", {}).get("recent_trades_n"), 20))

    income = _load_income_table(Path(pnl_csv))
    income_window, start_ts, end_ts = _slice_window(income, window_hours=window_hours)
    equity = _load_equity_table(Path(equity_csv))
    baseline = _load_baseline_bundle(cfg)

    portfolio_df, realized = _portfolio_summary(
        df_window=income_window,
        df_equity=equity,
        start_ts=start_ts,
        end_ts=end_ts,
        cfg=cfg,
        run_ts=run_ts,
    )
    portfolio_pnl = _safe_float(portfolio_df.iloc[0]["pnl_net"], 0.0) if not portfolio_df.empty else 0.0

    symbol_df = _symbol_summary(
        df_window=income_window,
        realized=realized,
        portfolio_pnl_net=portfolio_pnl,
        recent_trades_n=recent_n,
        run_ts=run_ts,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    concentration_df = _concentration_summary(
        symbol_df=symbol_df,
        cfg=cfg,
        run_ts=run_ts,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    drift_df = _drift_flags(
        portfolio_df=portfolio_df,
        symbol_df=symbol_df,
        concentration_df=concentration_df,
        baseline=baseline,
        cfg=cfg,
        run_ts=run_ts,
    )

    execution_df = _execution_quality_summary(
        cfg=cfg,
        run_ts=run_ts,
        start_ts=start_ts,
        end_ts=end_ts,
        order_register_csv=Path(order_register_csv),
        order_submit_log_csv=Path(order_submit_log_csv),
        order_lifecycle_log_csv=Path(order_lifecycle_log_csv),
        execution_ledger_csv=Path(execution_ledger_csv),
    )

    alerts_df = _alerts(
        portfolio_df=portfolio_df,
        concentration_df=concentration_df,
        drift_df=drift_df,
        execution_df=execution_df,
        cfg=cfg,
        run_ts=run_ts,
    )

    human = _human_summary(
        portfolio_df=portfolio_df,
        concentration_df=concentration_df,
        drift_df=drift_df,
        symbol_df=symbol_df,
    )

    append_mode = bool(cfg.get("outputs", {}).get("append", True))
    out = Path(out_dir)
    files = {
        "portfolio_summary": str(out / "paper_live_portfolio_summary.csv"),
        "symbol_summary": str(out / "paper_live_symbol_summary.csv"),
        "concentration": str(out / "paper_live_concentration.csv"),
        "drift_flags": str(out / "paper_live_drift_flags.csv"),
        "execution_quality": str(out / "paper_live_execution_quality.csv"),
        "alerts": str(out / "paper_live_alerts.csv"),
        "summary_json": str(out / "paper_live_summary.json"),
    }

    _append_csv(portfolio_df, Path(files["portfolio_summary"]), append=append_mode)
    _append_csv(symbol_df, Path(files["symbol_summary"]), append=append_mode)
    _append_csv(concentration_df, Path(files["concentration"]), append=append_mode)
    _append_csv(drift_df, Path(files["drift_flags"]), append=append_mode)
    _append_csv(execution_df, Path(files["execution_quality"]), append=append_mode)
    _append_csv(alerts_df, Path(files["alerts"]), append=append_mode)

    summary = {
        "run_ts_utc": _fmt_ts(run_ts),
        "window_hours": window_hours,
        "window_start_utc": _fmt_ts(start_ts),
        "window_end_utc": _fmt_ts(end_ts),
        "input_files": {
            "pnl_csv": str(Path(pnl_csv)),
            "equity_csv": str(Path(equity_csv)),
            "order_register_csv": str(Path(order_register_csv)),
            "order_submit_log_csv": str(Path(order_submit_log_csv)),
            "order_lifecycle_log_csv": str(Path(order_lifecycle_log_csv)),
            "execution_ledger_csv": str(Path(execution_ledger_csv)),
            "config_path": str(Path(config_path)) if config_path is not None else None,
        },
        "portfolio": portfolio_df.to_dict(orient="records"),
        "concentration": concentration_df.to_dict(orient="records"),
        "drift_flags": drift_df.to_dict(orient="records"),
        "alerts": alerts_df.to_dict(orient="records"),
        "human_summary": human,
        "baseline_reference": {
            "profile": baseline.get("profile", {}),
            "concentration": baseline.get("concentration", {}),
        },
        "output_files": files,
    }

    out.mkdir(parents=True, exist_ok=True)
    Path(files["summary_json"]).write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    return MonitorResult(
        portfolio=portfolio_df,
        symbol=symbol_df,
        concentration=concentration_df,
        drift_flags=drift_df,
        execution_quality=execution_df,
        alerts=alerts_df,
        summary=summary,
        output_files=files,
    )
