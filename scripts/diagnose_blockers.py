#!/usr/bin/env python3
"""diagnose_blockers.py — ¿qué gate concreto mantiene callado a cada par?

Para cada símbolo descompone la señal de `indicadores._calc_symbol` en sus
condiciones individuales y reporta, sobre la ventana analizada:

  - fail%: porcentaje de barras donde la condición falla.
  - near-miss: barras donde TODAS las demás condiciones pasan y solo esa
    falla — el blocker marginal. Es la métrica accionable: relajar ese gate
    convertiría exactamente esas barras en señal.

La descomposición se valida contra Long_Signal/Short_Signal reales: si el
AND de las condiciones no reproduce la señal del módulo, el script aborta.

Uso:
  python3 scripts/diagnose_blockers.py --days 30
  python3 scripts/diagnose_blockers.py --symbols CFX-USDT,ETH-USDT --days 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg import indicadores as ind  # noqa: E402
from pkg.backtesting import load_candles  # noqa: E402

WARMUP_DAYS = 15  # margen para EMAs/ADX/VOL_MA antes de la ventana analizada


def _true(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def decompose(df: pd.DataFrame, p: dict, side: str) -> dict:
    """Reconstruye las condiciones de señal (espejo de _calc_symbol)."""
    long_side = side == "long"
    c = df["close"]

    adx_min = float(p.get("adx_min", 15))
    long_adx_min = float(p.get("long_adx_min", adx_min))
    min_atr = float(p.get("min_atr_pct", 0.0012))
    max_atr = float(p.get("max_atr_pct", 0.012))
    require_close_vs_emas = bool(p.get("require_close_vs_emas", True))
    min_ema_spread = float(p.get("min_ema_spread", 0.001))
    min_vol_ratio = float(p.get("min_vol_ratio", 1.1))
    long_min_vol_ratio = float(p.get("long_min_vol_ratio", min_vol_ratio))
    adx_slope_min = float(p.get("adx_slope_min", 0.4))
    rsi_buy = int(p.get("rsi_buy", 55))
    rsi_sell = int(p.get("rsi_sell", 45))
    logic = str(p.get("logic", "any"))
    hhll_n = int(p.get("hhll_lookback", 10) or 0)
    fresh_breakout_only = bool(p.get("fresh_breakout_only", False))
    require_rsi_cross = bool(p.get("require_rsi_cross", True))
    long_require_rsi_and_breakout = bool(p.get("long_require_rsi_and_breakout", False))

    conds = {}
    if long_side:
        conds["trend_ema"] = df["EMA_S"] > df["EMA_L"]
        conds["adx_min"] = df["ADX"] >= long_adx_min
    else:
        conds["trend_ema"] = df["EMA_S"] < df["EMA_L"]
        conds["adx_min"] = df["ADX"] >= adx_min
    conds["atr_window"] = (df["ATR_pct"] >= min_atr) & (df["ATR_pct"] <= max_atr)
    conds["dist_emaslow"] = df["DIST_OK"] if isinstance(df["DIST_OK"], pd.Series) else _true(df)
    if require_close_vs_emas:
        conds["price_vs_emas"] = df["PRICE_LONG_OK"] if long_side else df["PRICE_SHORT_OK"]
    else:
        conds["price_vs_emas"] = _true(df)
    conds["ema_spread"] = df["EMA_SPREAD_OK"] if min_ema_spread > 0 else _true(df)
    vol_col = "VOL_OK_LONG" if long_side else "VOL_OK_SHORT"
    ratio = long_min_vol_ratio if long_side else min_vol_ratio
    conds["vol_ratio"] = df[vol_col] if ratio > 0 else _true(df)
    conds["adx_slope"] = df["ADX_SLOPE_OK"] if adx_slope_min > 0 else _true(df)

    # Trigger: momentum (RSI/breakout) y recencia de cruce EMA por separado
    if long_side:
        rsi_plain = df["RSI"] >= rsi_buy
        rsi_trig = df["RSI_LONG_RECENT"] if require_rsi_cross else rsi_plain
        if hhll_n > 0:
            brk = (c > df["HH"]) if not fresh_breakout_only else df["FRESH_LONG_BREAK"]
        else:
            brk = pd.Series(False, index=df.index)
        if logic == "strict" or long_require_rsi_and_breakout:
            conds["momentum_trigger"] = rsi_trig & brk
        else:
            conds["momentum_trigger"] = rsi_trig | brk
        conds["ema_cross_recent"] = df["EMA_LONG_RECENT"]
    else:
        rsi_plain = df["RSI"] <= rsi_sell
        rsi_trig = df["RSI_SHORT_RECENT"] if require_rsi_cross else rsi_plain
        if hhll_n > 0:
            brk = (c < df["LL"]) if not fresh_breakout_only else df["FRESH_SHORT_BREAK"]
        else:
            brk = pd.Series(False, index=df.index)
        if logic == "strict":
            conds["momentum_trigger"] = rsi_trig & brk
        else:
            conds["momentum_trigger"] = rsi_trig | brk
        conds["ema_cross_recent"] = df["EMA_SHORT_RECENT"]

    conds["no_funding_window"] = ~df["FUNDING_WINDOW"].astype(bool)
    return {k: v.fillna(False).astype(bool) for k, v in conds.items()}


def analyze_symbol(sym: str, params: dict, data_template: str, days: int):
    df = load_candles(data_template, sym, lookback_days=days + WARMUP_DAYS)
    di = ind._calc_symbol(df.copy(), sym, params_override=params)
    di["date"] = pd.to_datetime(di["date"], utc=True, errors="coerce")
    cutoff = di["date"].max() - pd.Timedelta(days=days)
    win = di[di["date"] >= cutoff].copy()

    out = {}
    for side, sig_col in (("long", "Long_Signal"), ("short", "Short_Signal")):
        conds = decompose(win, params, side)
        all_pass = pd.Series(True, index=win.index)
        for v in conds.values():
            all_pass &= v
        actual = win[sig_col].fillna(False).astype(bool)
        if not (all_pass == actual).all():
            mism = int((all_pass != actual).sum())
            raise RuntimeError(f"{sym} {side}: descomposición no reproduce la señal ({mism} barras)")

        n = len(win)
        signals = int(actual.sum())
        rows = []
        for name, v in conds.items():
            others = pd.Series(True, index=win.index)
            for k2, v2 in conds.items():
                if k2 != name:
                    others &= v2
            near_miss = int((others & ~v).sum())
            rows.append({
                "condition": name,
                "fail_pct": round(float((~v).mean() * 100), 1),
                "near_miss": near_miss,
            })
        out[side] = {"bars": n, "signals": signals, "conds": rows}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best_prod", default="pkg/best_prod.json")
    ap.add_argument("--data_template", default="archivos/cripto_price_5m_long.csv")
    ap.add_argument("--symbols", default="", help="Lista separada por comas (default: todos los de best_prod)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    best = json.loads((REPO_ROOT / args.best_prod).read_text(encoding="utf-8"))
    params_map = {e["symbol"]: e.get("params", {}) for e in best}
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or sorted(params_map)

    all_rows = []
    for sym in symbols:
        p = params_map.get(sym)
        if not p:
            print(f"⚠️ {sym}: sin params en best_prod, salto")
            continue
        res = analyze_symbol(sym, p, args.data_template, args.days)
        for side in ("long", "short"):
            r = res[side]
            print(f"\n=== {sym} {side.upper()} — {r['signals']} señales en {r['bars']} barras ({args.days}d) ===")
            ordered = sorted(r["conds"], key=lambda x: -x["near_miss"])
            print(f"  {'condición':<20} {'fail%':>7} {'near-miss':>10}")
            for row in ordered:
                mark = " ←" if row["near_miss"] > 0 and row is ordered[0] and row["near_miss"] >= 5 else ""
                print(f"  {row['condition']:<20} {row['fail_pct']:>6.1f}% {row['near_miss']:>10}{mark}")
                all_rows.append({"symbol": sym, "side": side, **row,
                                 "signals": r["signals"], "bars": r["bars"]})

    if args.out_csv:
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
        print(f"\nCSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
