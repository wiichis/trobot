#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg import backtesting


@dataclass
class EvalWindow:
    days: int
    trades: int
    pnl_net: float
    cost_ratio: Optional[float]
    max_dd_pct: Optional[float]
    winrate_pct: float


@dataclass
class EvalResult:
    best_path: str
    symbols: int
    short: EvalWindow
    long: EvalWindow


def _load_symbols(best_path: Path) -> List[str]:
    entries = backtesting._load_best_params(str(best_path))
    symbols = []
    for e in entries:
        sym = str(e.get("symbol", "")).upper().strip()
        if sym:
            symbols.append(sym)
    return sorted(set(symbols))


def _evaluate(best_path: Path, data_template: str, capital: float, short_days: int, long_days: int) -> EvalResult:
    symbols = _load_symbols(best_path)
    if not symbols:
        raise ValueError(f"No symbols found in best file: {best_path}")

    short_res = backtesting.run_live_parity_portfolio(
        symbols=symbols,
        data_template=data_template,
        capital=capital,
        weights_csv=None,
        best_path=str(best_path),
        lookback_days=short_days,
    )
    long_res = backtesting.run_live_parity_portfolio(
        symbols=symbols,
        data_template=data_template,
        capital=capital,
        weights_csv=None,
        best_path=str(best_path),
        lookback_days=long_days,
    )
    return EvalResult(
        best_path=str(best_path),
        symbols=len(symbols),
        short=EvalWindow(
            days=short_days,
            trades=int(short_res.get("trades", 0) or 0),
            pnl_net=float(short_res.get("pnl_net", 0.0) or 0.0),
            cost_ratio=(
                None
                if short_res.get("cost_ratio") is None
                else float(short_res.get("cost_ratio"))
            ),
            max_dd_pct=(
                None
                if short_res.get("max_dd_pct") is None
                else float(short_res.get("max_dd_pct"))
            ),
            winrate_pct=float(short_res.get("winrate_pct", 0.0) or 0.0),
        ),
        long=EvalWindow(
            days=long_days,
            trades=int(long_res.get("trades", 0) or 0),
            pnl_net=float(long_res.get("pnl_net", 0.0) or 0.0),
            cost_ratio=(
                None
                if long_res.get("cost_ratio") is None
                else float(long_res.get("cost_ratio"))
            ),
            max_dd_pct=(
                None
                if long_res.get("max_dd_pct") is None
                else float(long_res.get("max_dd_pct"))
            ),
            winrate_pct=float(long_res.get("winrate_pct", 0.0) or 0.0),
        ),
    )


def _check_gates(
    candidate: EvalResult,
    current: Optional[EvalResult],
    min_trades_short: int,
    min_trades_long: int,
    max_cost_ratio: float,
    max_dd_abs: float,
    min_pnl_short: float,
    min_pnl_long: float,
    allowed_pnl_drop_pct: float,
) -> List[str]:
    fails = []

    if candidate.short.trades < min_trades_short:
        fails.append(
            f"short trades {candidate.short.trades} < min {min_trades_short}"
        )
    if candidate.long.trades < min_trades_long:
        fails.append(
            f"long trades {candidate.long.trades} < min {min_trades_long}"
        )
    if candidate.short.pnl_net < min_pnl_short:
        fails.append(
            f"short pnl {candidate.short.pnl_net:.2f} < min {min_pnl_short:.2f}"
        )
    if candidate.long.pnl_net < min_pnl_long:
        fails.append(
            f"long pnl {candidate.long.pnl_net:.2f} < min {min_pnl_long:.2f}"
        )

    if candidate.short.cost_ratio is None:
        fails.append("short cost_ratio is None")
    elif candidate.short.cost_ratio > max_cost_ratio:
        fails.append(
            f"short cost_ratio {candidate.short.cost_ratio:.4f} > max {max_cost_ratio:.4f}"
        )
    if candidate.long.cost_ratio is None:
        fails.append("long cost_ratio is None")
    elif candidate.long.cost_ratio > max_cost_ratio:
        fails.append(
            f"long cost_ratio {candidate.long.cost_ratio:.4f} > max {max_cost_ratio:.4f}"
        )

    if candidate.short.max_dd_pct is None:
        fails.append("short max_dd_pct is None")
    elif abs(candidate.short.max_dd_pct) > max_dd_abs:
        fails.append(
            f"short |max_dd| {abs(candidate.short.max_dd_pct):.4f} > max {max_dd_abs:.4f}"
        )
    if candidate.long.max_dd_pct is None:
        fails.append("long max_dd_pct is None")
    elif abs(candidate.long.max_dd_pct) > max_dd_abs:
        fails.append(
            f"long |max_dd| {abs(candidate.long.max_dd_pct):.4f} > max {max_dd_abs:.4f}"
        )

    if current is not None:
        for label, cand_pnl, cur_pnl in [
            ("short", candidate.short.pnl_net, current.short.pnl_net),
            ("long", candidate.long.pnl_net, current.long.pnl_net),
        ]:
            if cur_pnl > 0:
                floor = cur_pnl * (1.0 - allowed_pnl_drop_pct)
                if cand_pnl < floor:
                    fails.append(
                        f"{label} pnl {cand_pnl:.2f} below allowed floor {floor:.2f} "
                        f"(current={cur_pnl:.2f}, drop={allowed_pnl_drop_pct:.2%})"
                    )
            else:
                if cand_pnl <= cur_pnl:
                    fails.append(
                        f"{label} pnl {cand_pnl:.2f} not better than current {cur_pnl:.2f}"
                    )

    return fails


def _backup(path: Path, backup_dir: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{path.stem}_{ts}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate best file and promote to production if it passes gates."
    )
    parser.add_argument(
        "--candidate_best",
        default="archivos/backtesting/daily/best_prod_daily_consistent.json",
        help="Candidate best JSON to evaluate.",
    )
    parser.add_argument(
        "--current_best",
        default="pkg/best_prod_consistent.json",
        help="Current production best JSON.",
    )
    parser.add_argument(
        "--target_best",
        default="pkg/best_prod_consistent.json",
        help="Destination path when candidate is promoted.",
    )
    parser.add_argument(
        "--target_fallback",
        default="pkg/best_prod.json",
        help="Optional second destination for compatibility.",
    )
    parser.add_argument(
        "--backup_dir",
        default="archivos/backtesting/daily/backups",
        help="Directory for backups before promotion.",
    )
    parser.add_argument(
        "--data_template",
        default="archivos/cripto_price_5m_long.csv",
        help="CSV template/path used for parity evaluation.",
    )
    parser.add_argument("--capital", type=float, default=300.0)
    parser.add_argument("--short_days", type=int, default=90)
    parser.add_argument("--long_days", type=int, default=180)
    parser.add_argument("--min_trades_short", type=int, default=20)
    parser.add_argument("--min_trades_long", type=int, default=35)
    parser.add_argument("--max_cost_ratio", type=float, default=1.0)
    parser.add_argument("--max_dd_abs", type=float, default=0.12)
    parser.add_argument("--min_pnl_short", type=float, default=0.0)
    parser.add_argument("--min_pnl_long", type=float, default=0.0)
    parser.add_argument(
        "--allowed_pnl_drop_pct",
        type=float,
        default=0.10,
        help="Allowed degradation vs current model (0.10 = up to 10%% drop).",
    )
    parser.add_argument(
        "--report",
        default="archivos/backtesting/daily/promotion_report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--refresh_indicators",
        action="store_true",
        help="Rebuild indicadores.csv after promotion.",
    )
    args = parser.parse_args()

    candidate_path = Path(args.candidate_best)
    current_path = Path(args.current_best)
    target_path = Path(args.target_best)
    fallback_path = Path(args.target_fallback) if args.target_fallback else None
    report_path = Path(args.report)
    backup_dir = Path(args.backup_dir)

    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate best file not found: {candidate_path}")

    candidate_eval = _evaluate(
        candidate_path,
        args.data_template,
        args.capital,
        args.short_days,
        args.long_days,
    )

    current_eval = None
    if current_path.exists():
        current_eval = _evaluate(
            current_path,
            args.data_template,
            args.capital,
            args.short_days,
            args.long_days,
        )

    fails = _check_gates(
        candidate=candidate_eval,
        current=current_eval,
        min_trades_short=args.min_trades_short,
        min_trades_long=args.min_trades_long,
        max_cost_ratio=args.max_cost_ratio,
        max_dd_abs=args.max_dd_abs,
        min_pnl_short=args.min_pnl_short,
        min_pnl_long=args.min_pnl_long,
        allowed_pnl_drop_pct=args.allowed_pnl_drop_pct,
    )

    promoted = len(fails) == 0
    backups: Dict[str, Optional[str]] = {"target_best": None, "target_fallback": None}
    if promoted:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        backups["target_best"] = str(_backup(target_path, backup_dir)) if target_path.exists() else None
        shutil.copy2(candidate_path, target_path)
        if fallback_path:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            backups["target_fallback"] = (
                str(_backup(fallback_path, backup_dir)) if fallback_path.exists() else None
            )
            shutil.copy2(candidate_path, fallback_path)
        if args.refresh_indicators:
            import pkg.indicadores as indicadores

            indicadores.update_indicators()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": asdict(candidate_eval),
        "current": asdict(current_eval) if current_eval else None,
        "gates_failed": fails,
        "promoted": promoted,
        "target_best": str(target_path),
        "target_fallback": str(fallback_path) if fallback_path else None,
        "backups": backups,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PROMOTION] candidate={candidate_path}")
    print(
        f"[PROMOTION] short pnl={candidate_eval.short.pnl_net:.2f} trades={candidate_eval.short.trades} "
        f"cost_ratio={candidate_eval.short.cost_ratio} dd={candidate_eval.short.max_dd_pct}"
    )
    print(
        f"[PROMOTION] long  pnl={candidate_eval.long.pnl_net:.2f} trades={candidate_eval.long.trades} "
        f"cost_ratio={candidate_eval.long.cost_ratio} dd={candidate_eval.long.max_dd_pct}"
    )
    if promoted:
        print(f"[PROMOTION] promoted -> {target_path}")
        if fallback_path:
            print(f"[PROMOTION] fallback synced -> {fallback_path}")
    else:
        print("[PROMOTION] not promoted. gates failed:")
        for f in fails:
            print(f"  - {f}")
    print(f"[PROMOTION] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
