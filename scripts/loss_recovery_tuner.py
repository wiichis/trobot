#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg import backtesting  # noqa: E402


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_best_entries(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Best file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Invalid best file format (expected list): {path}")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        params = item.get("params", {})
        if not symbol or not isinstance(params, dict):
            continue
        out.append({"symbol": symbol, "params": params})
    return out


def _entries_to_map(entries: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    data = {}
    for e in entries:
        sym = str(e["symbol"]).upper().strip()
        data[sym] = e
    return data


def _parse_income_table(
    pnl_csv: Path,
    lookback_hours: int,
    income_types: List[str],
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if not pnl_csv.exists():
        raise FileNotFoundError(f"PnL file not found: {pnl_csv}")
    df = pd.read_csv(pnl_csv)
    required = {"symbol", "income", "incomeType", "time"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"PnL file missing columns: {missing}")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "income", "time"])

    if income_types:
        allow = set([x.strip().upper() for x in income_types if x.strip()])
        df = df[df["incomeType"].astype(str).str.upper().isin(allow)]

    if df.empty:
        raise ValueError("PnL table is empty after filters.")

    end_ts = df["time"].max()
    start_ts = end_ts - pd.Timedelta(hours=int(lookback_hours))
    recent = df[df["time"] >= start_ts].copy()
    if recent.empty:
        raise ValueError("No rows in selected lookback window.")

    grouped = (
        recent.groupby("symbol", as_index=False)
        .agg(
            pnl_window=("income", "sum"),
            rows=("income", "size"),
        )
        .sort_values("pnl_window", ascending=True)
        .reset_index(drop=True)
    )
    return grouped, start_ts, end_ts


def _load_cooldown(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k.upper().strip()] = v
    return out


def _save_cooldown(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _is_in_cooldown(last_ts_iso: Optional[str], cooldown_hours: int, now_utc: datetime) -> bool:
    if not last_ts_iso:
        return False
    try:
        last_dt = datetime.fromisoformat(last_ts_iso)
    except Exception:
        return False
    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)
    return now_utc < (last_dt + timedelta(hours=int(cooldown_hours)))


def _run_backtest_for_symbols(
    symbols: List[str],
    data_template: str,
    sweep_cfg: str,
    out_dir: Path,
    lookback_days: int,
    n_trials: int,
    min_trades: int,
    max_daily_sl_streak: int,
    max_cost_ratio: float,
    max_dd: float,
    rank_by: str,
    candidate_best_path: Path,
) -> None:
    cmd = [
        sys.executable,
        "pkg/backtesting.py",
        "--symbols",
        ",".join(symbols),
        "--data_template",
        data_template,
        "--lookback_days",
        str(int(lookback_days)),
        "--train_ratio",
        "0",
        "--search_mode",
        "random",
        "--n_trials",
        str(int(n_trials)),
        "--rank_by",
        rank_by,
        "--sweep",
        sweep_cfg,
        "--out_dir",
        str(out_dir),
        "--min_trades",
        str(int(min_trades)),
        "--max_daily_sl_streak",
        str(int(max_daily_sl_streak)),
        "--max_cost_ratio",
        str(float(max_cost_ratio)),
        "--max_dd",
        str(float(max_dd)),
        "--export_best",
        str(candidate_best_path),
        "--export_positive_ratio",
    ]
    print("[RECOVERY] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _eval_symbol(best_path: Path, symbol: str, data_template: str, capital: float, days: int) -> Dict[str, object]:
    res = backtesting.run_live_parity_portfolio(
        symbols=[symbol],
        data_template=data_template,
        capital=float(capital),
        weights_csv=None,
        best_path=str(best_path),
        lookback_days=int(days),
    )
    return {
        "trades": int(res.get("trades", 0) or 0),
        "pnl_net": float(res.get("pnl_net", 0.0) or 0.0),
        "cost_ratio": None if res.get("cost_ratio") is None else float(res.get("cost_ratio")),
        "max_dd_pct": None if res.get("max_dd_pct") is None else float(res.get("max_dd_pct")),
        "winrate_pct": float(res.get("winrate_pct", 0.0) or 0.0),
    }


def _candidate_passes(
    old_short: Dict[str, object],
    new_short: Dict[str, object],
    old_long: Dict[str, object],
    new_long: Dict[str, object],
    min_trades_short: int,
    min_short_pnl_improvement: float,
    max_cost_ratio: float,
    max_dd_abs: float,
    allowed_long_drop_pct: float,
) -> Tuple[bool, List[str]]:
    reasons = []

    if int(new_short["trades"]) < int(min_trades_short):
        reasons.append(f"short trades {new_short['trades']} < {min_trades_short}")

    improve = float(new_short["pnl_net"]) - float(old_short["pnl_net"])
    if improve < float(min_short_pnl_improvement):
        reasons.append(
            f"short pnl improvement {improve:.4f} < {float(min_short_pnl_improvement):.4f}"
        )

    for label, data in [("short", new_short), ("long", new_long)]:
        cr = data["cost_ratio"]
        if cr is None:
            reasons.append(f"{label} cost_ratio is None")
        elif float(cr) > float(max_cost_ratio):
            reasons.append(f"{label} cost_ratio {float(cr):.4f} > {float(max_cost_ratio):.4f}")

        dd = data["max_dd_pct"]
        if dd is None:
            reasons.append(f"{label} max_dd_pct is None")
        elif abs(float(dd)) > abs(float(max_dd_abs)):
            reasons.append(f"{label} |max_dd| {abs(float(dd)):.4f} > {abs(float(max_dd_abs)):.4f}")

    old_long_pnl = float(old_long["pnl_net"])
    new_long_pnl = float(new_long["pnl_net"])
    if old_long_pnl > 0:
        floor = old_long_pnl * (1.0 - float(allowed_long_drop_pct))
        if new_long_pnl < floor:
            reasons.append(f"long pnl {new_long_pnl:.4f} < allowed floor {floor:.4f}")
    else:
        if new_long_pnl < old_long_pnl:
            reasons.append(f"long pnl {new_long_pnl:.4f} worse than old {old_long_pnl:.4f}")

    return (len(reasons) == 0), reasons


def _send_telegram_if_possible(message: str) -> Tuple[bool, str]:
    try:
        import pkg.credentials as credentials
    except Exception as exc:
        return False, f"credentials import failed: {exc}"

    token = getattr(credentials, "token", None)
    chat_id = getattr(credentials, "chatID", None) or getattr(credentials, "chat_id", None)
    if not token or not chat_id:
        return False, "missing token/chat_id in pkg.credentials"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, data=payload, timeout=12)
    except Exception as exc:
        return False, f"telegram request failed: {exc}"
    if not resp.ok:
        body = resp.text[:220] if resp.text else ""
        return False, f"telegram status={resp.status_code} body={body}"
    return True, "ok"


def _format_alert_message(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    args,
    candidates_raw: List[Dict[str, object]],
    blocked: List[str],
    eligible_symbols: List[str],
    promoted: List[str],
    rejected: List[Dict[str, str]],
    tuned: List[Dict[str, object]],
    applied: bool,
    report_path: Path,
) -> str:
    lines = []
    lines.append("🤖 TRobot | Ajuste por perdidas")
    lines.append(f"🕒 Ventana UTC: {start_ts} → {end_ts}")
    lines.append(
        f"🎯 Regla: pnl_24h <= {float(args.loss_trigger):.2f} | ⏳ Cooldown: {int(args.cooldown_hours)}h"
    )
    lines.append("")
    lines.append("📌 Resumen")
    lines.append(f"• Monedas candidatas: {len(candidates_raw)}")
    lines.append(f"• En cooldown: {len(blocked)}")
    lines.append(f"• Evaluadas: {len(eligible_symbols)}")
    lines.append(f"• Promovidas: {len(promoted)} ({', '.join(promoted) if promoted else '-'})")
    lines.append(f"• Aplicado en produccion: {'✅ SI' if applied else '❌ NO'}")

    if candidates_raw:
        lines.append("")
        lines.append("📉 Mayores perdidas (24h)")
        for row in candidates_raw[:5]:
            lines.append(
                f"• {row['symbol']}: pnl={float(row['pnl_window']):.4f} | eventos={int(row['rows'])}"
            )

    if tuned:
        lines.append("")
        lines.append("🛠 Resultado del ajuste")
        for t in tuned[:5]:
            sym = t["symbol"]
            old_short = t["old_short"]
            new_short = t["new_short"]
            status = "✅ Pasa" if t["passes"] else "❌ No pasa"
            lines.append(
                f"• {sym}: pnl corto {float(old_short['pnl_net']):.4f} → {float(new_short['pnl_net']):.4f} | {status}"
            )

    if rejected:
        lines.append("")
        lines.append("🚫 Rechazadas")
        for r in rejected[:5]:
            reason = str(r.get("reason", ""))[:120]
            lines.append(f"• {r['symbol']}: {reason}")

    if blocked:
        lines.append("")
        lines.append(f"🧊 En cooldown: {', '.join(blocked[:8])}")

    lines.append("")
    lines.append(f"📄 Reporte: {report_path}")

    msg = "\n".join(lines)
    # Telegram hard limit is 4096 chars. Keep head section if we exceed.
    if len(msg) > 3900:
        msg = msg[:3850] + "\n...\nReport: " + str(report_path)
    return msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily loss-recovery tuner: optimize only symbols with recent losses."
    )
    parser.add_argument("--pnl_csv", default="archivos/PnL.csv")
    parser.add_argument("--best_consistent", default="pkg/best_prod_consistent.json")
    parser.add_argument("--best_fallback", default="pkg/best_prod.json")
    parser.add_argument("--data_template", default="archivos/cripto_price_5m_long.csv")
    parser.add_argument("--sweep_cfg", default="archivos/backtesting/simple_sweep.json")
    parser.add_argument("--out_dir", default="archivos/backtesting/daily")
    parser.add_argument("--lookback_hours", type=int, default=24)
    parser.add_argument(
        "--income_types",
        default="REALIZED_PNL,TRADING_FEE,FUNDING_FEE",
        help="Comma-separated incomeType values used to compute net daily pnl.",
    )
    parser.add_argument("--loss_trigger", type=float, default=-1.5, help="Trigger when pnl_window <= this value.")
    parser.add_argument("--max_symbols_per_run", type=int, default=3)
    parser.add_argument("--cooldown_hours", type=int, default=72)
    parser.add_argument("--cooldown_file", default="archivos/backtesting/daily/loss_recovery_cooldown.json")
    parser.add_argument("--lookback_days_opt", type=int, default=60)
    parser.add_argument("--lookback_days_long", type=int, default=120)
    parser.add_argument("--n_trials", type=int, default=45)
    parser.add_argument("--min_trades", type=int, default=4)
    parser.add_argument("--min_trades_short_eval", type=int, default=4)
    parser.add_argument("--max_daily_sl_streak", type=int, default=3)
    parser.add_argument("--max_cost_ratio_opt", type=float, default=0.95)
    parser.add_argument("--max_dd_opt", type=float, default=0.12)
    parser.add_argument("--max_cost_ratio_gate", type=float, default=1.0)
    parser.add_argument("--max_dd_gate", type=float, default=0.12)
    parser.add_argument("--min_short_pnl_improvement", type=float, default=0.0)
    parser.add_argument("--allowed_long_drop_pct", type=float, default=0.10)
    parser.add_argument("--rank_by", default="pnl_net_per_trade")
    parser.add_argument("--send_alert", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Apply promoted symbol params to production best files.")
    parser.add_argument("--refresh_indicators", action="store_true")
    parser.add_argument("--report", default="archivos/backtesting/daily/loss_recovery_report.json")
    args = parser.parse_args()

    now = _now_utc()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    cooldown_path = Path(args.cooldown_file)
    candidate_best = out_dir / "best_prod_loss_candidate.json"

    current_path = Path(args.best_consistent)
    fallback_path = Path(args.best_fallback)
    pnl_path = Path(args.pnl_csv)

    current_entries = _load_best_entries(current_path)
    current_symbols = [e["symbol"] for e in current_entries]
    current_map = _entries_to_map(current_entries)

    income_types = [x.strip().upper() for x in str(args.income_types).split(",") if x.strip()]
    pnl_grouped, start_ts, end_ts = _parse_income_table(
        pnl_path,
        lookback_hours=args.lookback_hours,
        income_types=income_types,
    )

    candidates_raw = []
    for _, row in pnl_grouped.iterrows():
        sym = str(row["symbol"]).upper()
        pnl_window = float(row["pnl_window"])
        if sym not in current_map:
            continue
        if pnl_window <= float(args.loss_trigger):
            candidates_raw.append(
                {
                    "symbol": sym,
                    "pnl_window": pnl_window,
                    "rows": int(row["rows"]),
                }
            )

    candidates_raw = sorted(candidates_raw, key=lambda x: x["pnl_window"])
    if int(args.max_symbols_per_run) > 0:
        candidates_raw = candidates_raw[: int(args.max_symbols_per_run)]

    cooldown = _load_cooldown(cooldown_path)
    blocked = []
    eligible_symbols = []
    for c in candidates_raw:
        sym = c["symbol"]
        if _is_in_cooldown(cooldown.get(sym), args.cooldown_hours, now):
            blocked.append(sym)
        else:
            eligible_symbols.append(sym)

    promoted = []
    rejected = []
    tuned = []
    backups = {}

    if eligible_symbols:
        _run_backtest_for_symbols(
            symbols=eligible_symbols,
            data_template=args.data_template,
            sweep_cfg=args.sweep_cfg,
            out_dir=out_dir,
            lookback_days=args.lookback_days_opt,
            n_trials=args.n_trials,
            min_trades=args.min_trades,
            max_daily_sl_streak=args.max_daily_sl_streak,
            max_cost_ratio=args.max_cost_ratio_opt,
            max_dd=args.max_dd_opt,
            rank_by=args.rank_by,
            candidate_best_path=candidate_best,
        )

        candidate_entries = _load_best_entries(candidate_best) if candidate_best.exists() else []
        candidate_map = _entries_to_map(candidate_entries)
        merged_map = dict(current_map)

        for sym in eligible_symbols:
            if sym not in candidate_map:
                rejected.append({"symbol": sym, "reason": "symbol not present in candidate best"})
                continue

            old_short = _eval_symbol(current_path, sym, args.data_template, 300.0, args.lookback_days_opt)
            new_short = _eval_symbol(candidate_best, sym, args.data_template, 300.0, args.lookback_days_opt)
            old_long = _eval_symbol(current_path, sym, args.data_template, 300.0, args.lookback_days_long)
            new_long = _eval_symbol(candidate_best, sym, args.data_template, 300.0, args.lookback_days_long)

            ok, reasons = _candidate_passes(
                old_short=old_short,
                new_short=new_short,
                old_long=old_long,
                new_long=new_long,
                min_trades_short=args.min_trades_short_eval,
                min_short_pnl_improvement=args.min_short_pnl_improvement,
                max_cost_ratio=args.max_cost_ratio_gate,
                max_dd_abs=args.max_dd_gate,
                allowed_long_drop_pct=args.allowed_long_drop_pct,
            )

            tuned.append(
                {
                    "symbol": sym,
                    "old_short": old_short,
                    "new_short": new_short,
                    "old_long": old_long,
                    "new_long": new_long,
                    "passes": ok,
                    "reasons": reasons,
                }
            )

            if not ok:
                rejected.append({"symbol": sym, "reason": "; ".join(reasons)})
                continue

            merged_map[sym] = candidate_map[sym]
            promoted.append(sym)

        merged_entries = [merged_map[k] for k in sorted(merged_map.keys())]
        merged_path = out_dir / "best_prod_consistent_loss_merged.json"
        merged_path.write_text(json.dumps(merged_entries, indent=2), encoding="utf-8")

        if args.apply and promoted:
            backup_dir = out_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = now.strftime("%Y%m%d_%H%M%S")
            backup_consistent = backup_dir / f"best_prod_consistent_{stamp}.json"
            shutil.copy2(current_path, backup_consistent)
            backups["best_consistent"] = str(backup_consistent)

            shutil.copy2(merged_path, current_path)
            if fallback_path:
                backup_fallback = backup_dir / f"best_prod_{stamp}.json"
                if fallback_path.exists():
                    shutil.copy2(fallback_path, backup_fallback)
                    backups["best_fallback"] = str(backup_fallback)
                shutil.copy2(merged_path, fallback_path)

            if args.refresh_indicators:
                import pkg.indicadores as indicadores

                indicadores.update_indicators()

        if args.apply:
            for sym in eligible_symbols:
                cooldown[sym] = now.isoformat()
            _save_cooldown(cooldown_path, cooldown)

    report = {
        "timestamp_utc": now.isoformat(),
        "window_start": str(start_ts),
        "window_end": str(end_ts),
        "lookback_hours": int(args.lookback_hours),
        "loss_trigger": float(args.loss_trigger),
        "current_symbols_count": len(current_symbols),
        "current_symbols": current_symbols,
        "candidates_raw": candidates_raw,
        "blocked_by_cooldown": blocked,
        "eligible_symbols": eligible_symbols,
        "tuned": tuned,
        "promoted": promoted,
        "rejected": rejected,
        "applied": bool(args.apply and len(promoted) > 0),
        "backups": backups,
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    applied = bool(args.apply and len(promoted) > 0)
    alert_message = _format_alert_message(
        start_ts=start_ts,
        end_ts=end_ts,
        args=args,
        candidates_raw=candidates_raw,
        blocked=blocked,
        eligible_symbols=eligible_symbols,
        promoted=promoted,
        rejected=rejected,
        tuned=tuned,
        applied=applied,
        report_path=report_path,
    )

    alert_status = {"enabled": bool(args.send_alert), "sent": False, "detail": "disabled"}
    if args.send_alert:
        ok, detail = _send_telegram_if_possible(alert_message)
        alert_status = {"enabled": True, "sent": ok, "detail": detail}
        print(f"[RECOVERY][ALERT] sent={ok} detail={detail}")
    report["alert"] = alert_status
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[RECOVERY] candidates_raw:", len(candidates_raw))
    print("[RECOVERY] blocked_by_cooldown:", len(blocked))
    print("[RECOVERY] eligible:", len(eligible_symbols))
    print("[RECOVERY] promoted:", len(promoted))
    print("[RECOVERY] applied:", bool(args.apply and len(promoted) > 0))
    print("[RECOVERY] report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
