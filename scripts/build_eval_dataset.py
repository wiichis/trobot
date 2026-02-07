#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def _require_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a clean evaluation dataset from long 5m candles."
    )
    parser.add_argument(
        "--source",
        default="archivos/cripto_price_5m_long.csv",
        help="Source long CSV path.",
    )
    parser.add_argument(
        "--output",
        default="archivos/cripto_price_5m_eval.csv",
        help="Output evaluation CSV path.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Trailing window in days for evaluation dataset.",
    )
    parser.add_argument(
        "--report",
        default="archivos/backtesting/daily/eval_dataset_report.json",
        help="Where to save dataset diagnostics in JSON.",
    )
    parser.add_argument(
        "--min_coverage_ratio",
        type=float,
        default=0.85,
        help="Warn if symbol coverage is below this ratio vs expected 5m bars.",
    )
    args = parser.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    report_path = Path(args.report)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    df = pd.read_csv(src)
    _require_columns(df, ["symbol", "date", "open", "high", "low", "close", "volume"])

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "date"])
    df = df.sort_values(["symbol", "date"])
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

    last_ts = df["date"].max()
    cutoff = last_ts - pd.Timedelta(days=int(args.days))
    df_eval = df[df["date"] >= cutoff].copy()
    df_eval = df_eval.sort_values(["symbol", "date"])

    out.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(out, index=False)

    expected_rows = int(args.days * 288)
    per_symbol = []
    for symbol, sdf in df_eval.groupby("symbol"):
        rows = int(len(sdf))
        first_ts = sdf["date"].min()
        last_ts_sym = sdf["date"].max()
        coverage_ratio = (rows / expected_rows) if expected_rows > 0 else 0.0
        per_symbol.append(
            {
                "symbol": symbol,
                "rows": rows,
                "first_date": str(first_ts),
                "last_date": str(last_ts_sym),
                "coverage_ratio": round(float(coverage_ratio), 4),
                "coverage_ok": bool(coverage_ratio >= float(args.min_coverage_ratio)),
            }
        )

    report = {
        "source": str(src),
        "output": str(out),
        "days": int(args.days),
        "cutoff": str(cutoff),
        "last_timestamp_source": str(last_ts),
        "rows_source": int(len(df)),
        "rows_eval": int(len(df_eval)),
        "symbols_eval": int(df_eval["symbol"].nunique()) if not df_eval.empty else 0,
        "expected_rows_per_symbol": expected_rows,
        "min_coverage_ratio": float(args.min_coverage_ratio),
        "symbols": per_symbol,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    weak = [r["symbol"] for r in per_symbol if not r["coverage_ok"]]
    print(
        f"[EVAL_DATA] rows={report['rows_eval']} symbols={report['symbols_eval']} "
        f"window_days={report['days']} output={out}"
    )
    if weak:
        print(
            f"[EVAL_DATA][WARN] coverage below {args.min_coverage_ratio} for: "
            + ", ".join(weak)
        )
    print(f"[EVAL_DATA] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
