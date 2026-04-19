"""Página en vivo: posiciones abiertas, TPs/SL activos, últimos eventos lifecycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard import loaders

st.set_page_config(page_title="En vivo — TRobot", page_icon="📡", layout="wide")

st.title("📡 Estado en vivo")
st.caption("Muestra el estado del bot al momento del último sync.")

# ---------------------------------------------------------------------------
# Posiciones abiertas (derivadas de tp_stage + sl_watch)
# ---------------------------------------------------------------------------

st.subheader("Posiciones activas")

tp_stage = loaders.load_tp_stage()
sl_watch = loaders.load_sl_watch()

if tp_stage.empty and sl_watch.empty:
    st.info("No hay posiciones activas detectadas.")
else:
    # Usamos tp_stage_state como fuente principal
    if not tp_stage.empty:
        cols_interes = [c for c in ["symbol", "position_side", "entry_price", "stage",
                                     "tp1", "tp2", "tp3", "sl", "last_update"]
                        if c in tp_stage.columns]
        if cols_interes:
            st.dataframe(tp_stage[cols_interes], use_container_width=True, hide_index=True)
        else:
            st.dataframe(tp_stage, use_container_width=True, hide_index=True)
    elif not sl_watch.empty:
        st.dataframe(sl_watch, use_container_width=True, hide_index=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Últimos eventos lifecycle
# ---------------------------------------------------------------------------

st.subheader("Últimos eventos del bot")

events = loaders.load_lifecycle()
if events.empty:
    st.info("Sin eventos lifecycle.")
else:
    c1, c2 = st.columns([1, 3])
    n = c1.number_input("Mostrar últimos N", min_value=10, max_value=500, value=50, step=10)

    categories_all = sorted(events["category"].unique().tolist()) if "category" in events.columns else []
    categories_filter = c2.multiselect("Filtrar categorías", categories_all, default=[])

    df = events.tail(1000).copy()
    if categories_filter:
        df = df[df["category"].isin(categories_filter)]
    df = df.tail(n).sort_values("ts_utc", ascending=False)

    # Render de tabla compacta
    if "ts_utc" in df.columns:
        df["hora"] = df["ts_utc"].dt.strftime("%m-%d %H:%M")

    show_cols = [c for c in ["hora", "category", "severity", "body", "telegram_sent"] if c in df.columns]
    show = df[show_cols].copy()
    if "body" in show.columns:
        # simplificar cuerpo para tabla
        show["body"] = show["body"].astype(str).str.replace("\n", " · ").str.slice(0, 80)
    st.dataframe(show, use_container_width=True, hide_index=True, height=400)

st.markdown("---")

# ---------------------------------------------------------------------------
# Submit log: entradas recientes y fallos
# ---------------------------------------------------------------------------

st.subheader("Últimas órdenes (execution_ledger)")

ledger = loaders.load_execution_ledger()
if ledger.empty:
    st.info("Sin execution_ledger.")
else:
    focus_types = ["order_submitted", "entry_order_filled",
                   "entry_order_canceled_or_expired", "order_submit_failed",
                   "stop_loss_hit", "tp1_filled", "tp2_filled", "tp3_filled"]
    df = ledger[ledger["event_type"].isin(focus_types)].tail(30).copy()
    if "ts_utc" in df.columns:
        df["hora"] = df["ts_utc"].dt.strftime("%m-%d %H:%M")
    show_cols = [c for c in ["hora", "event_type", "symbol", "position_side",
                             "order_type", "actual_fill_price", "stop_price",
                             "fill_qty", "raw_msg"]
                 if c in df.columns]
    show = df[show_cols].sort_values("hora" if "hora" in show_cols else df.columns[0],
                                       ascending=False)
    st.dataframe(show, use_container_width=True, hide_index=True, height=400)
