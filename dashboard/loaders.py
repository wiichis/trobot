"""Data loaders con caché de Streamlit.

Cada loader lee un CSV, normaliza tipos y devuelve un DataFrame listo para
visualizar. El caché se invalida manualmente desde la sidebar tras un sync.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVOS = REPO_ROOT / "archivos"


def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_pnl() -> pd.DataFrame:
    """PnL.csv — REALIZED_PNL, TRADING_FEE, FUNDING_FEE por trade."""
    df = _safe_read(ARCHIVOS / "PnL.csv")
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["income"] = pd.to_numeric(df["income"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_ganancias() -> pd.DataFrame:
    """ganancias.csv — snapshot horario de balance."""
    df = _safe_read(ARCHIVOS / "ganancias.csv")
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Forzar naive para comparación consistente con pd.Timestamp.utcnow().tz_localize(None)
    if pd.api.types.is_datetime64tz_dtype(df["date"]):
        df["date"] = df["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    df = df.dropna(subset=["date", "balance"]).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_lifecycle() -> pd.DataFrame:
    df = _safe_read(ARCHIVOS / "lifecycle_event_log.csv")
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc").reset_index(drop=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_execution_ledger() -> pd.DataFrame:
    df = _safe_read(ARCHIVOS / "execution_ledger.csv")
    if df.empty:
        return df
    for col in ("ts_utc", "submit_time_utc", "fill_time_utc"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    for col in ("intended_entry_price", "submitted_price", "actual_fill_price",
                "stop_price", "submit_qty", "fill_qty"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("ts_utc").reset_index(drop=True) if "ts_utc" in df.columns else df
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_trade_closed() -> pd.DataFrame:
    df = _safe_read(ARCHIVOS / "trade_closed_log.csv")
    if df.empty:
        return df
    for col in df.columns:
        if "time" in col.lower() or col.endswith("_utc"):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


@st.cache_data(ttl=60, show_spinner=False)
def load_sl_watch() -> pd.DataFrame:
    return _safe_read(ARCHIVOS / "sl_watch.csv")


@st.cache_data(ttl=60, show_spinner=False)
def load_tp_stage() -> pd.DataFrame:
    return _safe_read(ARCHIVOS / "tp_stage_state.csv")


@st.cache_data(ttl=60, show_spinner=False)
def load_indicadores() -> pd.DataFrame:
    return _safe_read(ARCHIVOS / "indicadores.csv")


@st.cache_data(ttl=60, show_spinner=False)
def load_best_prod() -> List[Dict]:
    path = REPO_ROOT / "pkg" / "best_prod.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def clear_all_caches() -> None:
    """Invalida todos los cachés. Llamar después de un sync."""
    for fn in (
        load_pnl, load_ganancias, load_lifecycle, load_execution_ledger,
        load_trade_closed, load_sl_watch, load_tp_stage, load_indicadores,
        load_best_prod,
    ):
        fn.clear()


# ---------------------------------------------------------------------------
# Agregaciones reutilizables
# ---------------------------------------------------------------------------

def trades_by_symbol(pnl_df: pd.DataFrame, days: Optional[int] = None) -> pd.DataFrame:
    """Agrega por símbolo: trades, pnl, winrate, fees, funding."""
    if pnl_df.empty:
        return pd.DataFrame()
    df = pnl_df.copy()
    if days is not None:
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
        df = df[df["time"] >= cutoff]

    realized = df[df["incomeType"] == "REALIZED_PNL"]
    fees = df[df["incomeType"] == "COMMISSION"] if "COMMISSION" in df["incomeType"].values else \
           df[df["incomeType"].astype(str).str.contains("FEE", case=False, na=False)]
    funding = df[df["incomeType"] == "FUNDING_FEE"]

    agg = realized.groupby("symbol").agg(
        trades=("income", "size"),
        pnl_realized=("income", "sum"),
        winrate=("income", lambda s: (s > 0).mean() * 100 if len(s) else 0),
        best=("income", "max"),
        worst=("income", "min"),
    ).reset_index()

    fees_by_sym = fees.groupby("symbol")["income"].sum().rename("fees") if not fees.empty else pd.Series(dtype=float, name="fees")
    funding_by_sym = funding.groupby("symbol")["income"].sum().rename("funding") if not funding.empty else pd.Series(dtype=float, name="funding")

    agg = agg.merge(fees_by_sym, on="symbol", how="left").merge(funding_by_sym, on="symbol", how="left")
    agg["fees"] = agg["fees"].fillna(0.0)
    agg["funding"] = agg["funding"].fillna(0.0)
    agg["pnl_net"] = agg["pnl_realized"] + agg["fees"] + agg["funding"]

    # profit factor: sum(wins) / abs(sum(losses))
    def _pf(group):
        wins = group[group > 0].sum()
        losses = abs(group[group < 0].sum())
        return wins / losses if losses > 0 else float("inf") if wins > 0 else 0

    pf_map = realized.groupby("symbol")["income"].apply(_pf).rename("profit_factor")
    agg = agg.merge(pf_map, on="symbol", how="left")

    return agg.sort_values("pnl_net", ascending=False).reset_index(drop=True)


def daily_pnl(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """PnL neto por día (REALIZED + FEES + FUNDING)."""
    if pnl_df.empty:
        return pd.DataFrame()
    df = pnl_df.copy()
    df["day"] = df["time"].dt.floor("D")
    daily = df.groupby("day")["income"].sum().reset_index(name="pnl_day")
    daily = daily.sort_values("day").reset_index(drop=True)
    daily["pnl_cum"] = daily["pnl_day"].cumsum()
    return daily


def pnl_summary(pnl_df: pd.DataFrame, days: int) -> Dict[str, float]:
    """Totales: pnl neto, trades, winrate, profit factor, fees+funding."""
    if pnl_df.empty:
        return {"pnl_net": 0.0, "trades": 0, "winrate": 0.0, "profit_factor": 0.0,
                "fees_funding": 0.0, "pnl_realized": 0.0}
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = pnl_df[pnl_df["time"] >= cutoff]
    realized = df[df["incomeType"] == "REALIZED_PNL"]["income"]
    others = df[df["incomeType"] != "REALIZED_PNL"]["income"]

    wins = realized[realized > 0].sum()
    losses = abs(realized[realized < 0].sum())
    pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)

    return {
        "pnl_realized": float(realized.sum()),
        "fees_funding": float(others.sum()),
        "pnl_net": float(realized.sum() + others.sum()),
        "trades": int(len(realized)),
        "winrate": float((realized > 0).mean() * 100) if len(realized) else 0.0,
        "profit_factor": float(pf) if pf != float("inf") else 99.0,
    }
