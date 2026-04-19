"""Página de análisis por par — drill-down con params, PnL, backtest vs real."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import loaders

st.set_page_config(page_title="Pares — TRobot", page_icon="🔬", layout="wide")

st.title("🔬 Análisis por Par")

pnl = loaders.load_pnl()
best = loaders.load_best_prod()

if not best:
    st.warning("best_prod.json no disponible.")
    st.stop()

best_map = {x["symbol"]: x.get("params", {}) for x in best}
symbols = sorted(best_map.keys())

symbol = st.selectbox("Selecciona par", symbols)
params = best_map.get(symbol, {})

# ---------------------------------------------------------------------------
# KPIs del par en 3 ventanas
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader(f"{symbol} — PnL por ventana")

cols = st.columns(3)
for i, days in enumerate((7, 30, 90)):
    df = pnl[(pnl["symbol"] == symbol) & (pnl["time"] >= pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days))]
    realized = df[df["incomeType"] == "REALIZED_PNL"]
    fees_funding = df[df["incomeType"] != "REALIZED_PNL"]
    pnl_net = realized["income"].sum() + fees_funding["income"].sum()
    trades = len(realized)
    wr = (realized["income"] > 0).mean() * 100 if trades else 0
    emoji = "🟢" if pnl_net >= 0 else "🔴"
    with cols[i]:
        st.metric(f"{emoji} {days} días", f"{pnl_net:+.2f} USD",
                  f"{trades} trades · WR {wr:.0f}%")

# ---------------------------------------------------------------------------
# Gráfico PnL acumulado del par
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("PnL acumulado (90 días)")

window = pnl[(pnl["symbol"] == symbol) &
             (pnl["time"] >= pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=90))].copy()

if not window.empty:
    window = window.sort_values("time")
    window["pnl_cum"] = window["income"].cumsum()
    fig = px.area(window, x="time", y="pnl_cum", title=None)
    fig.update_traces(line=dict(color="#3498db", width=2),
                      fillcolor="rgba(52, 152, 219, 0.15)")
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                      yaxis_title="PnL acum (USD)", xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sin actividad en 90 días.")

# ---------------------------------------------------------------------------
# Distribución de trades del par
# ---------------------------------------------------------------------------

realized_90 = window[window["incomeType"] == "REALIZED_PNL"]
if not realized_90.empty:
    c1, c2 = st.columns(2)
    fig_hist = px.histogram(realized_90, x="income", nbins=20,
                             color=realized_90["income"].apply(lambda x: "Win" if x > 0 else "Loss"),
                             color_discrete_map={"Win": "#2ecc71", "Loss": "#e74c3c"},
                             title="Distribución de PnL por trade (90d)")
    fig_hist.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
    c1.plotly_chart(fig_hist, use_container_width=True)

    realized_90_sorted = realized_90.sort_values("time")
    fig_ts = px.scatter(
        realized_90_sorted, x="time", y="income",
        color=realized_90_sorted["income"].apply(lambda x: "Win" if x > 0 else "Loss"),
        color_discrete_map={"Win": "#2ecc71", "Loss": "#e74c3c"},
        title="Trades en el tiempo",
    )
    fig_ts.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
    fig_ts.add_hline(y=0, line_dash="dash", line_color="gray")
    c2.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Parámetros activos del par
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader(f"Parámetros activos ({symbol})")

if params:
    # Ordenar en categorías
    entry_keys = ["logic", "ema_fast", "ema_slow", "rsi_buy", "rsi_sell",
                  "adx_min", "min_atr_pct", "max_atr_pct", "max_dist_emaslow",
                  "fresh_cross_max_bars", "require_rsi_cross", "fresh_breakout_only",
                  "min_ema_spread", "min_vol_ratio", "vol_ma_len", "adx_slope_len",
                  "adx_slope_min", "require_close_vs_emas", "hhll_lookback"]
    tp_sl_keys = ["tp", "tp_mode", "tp_atr_mult", "sl_mode", "sl_pct", "atr_mult",
                  "be_trigger", "cooldown", "time_exit_bars"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Entrada**")
        entry = {k: params[k] for k in entry_keys if k in params}
        st.json(entry, expanded=True)
    with c2:
        st.markdown("**TP / SL / Gestión**")
        tp = {k: params[k] for k in tp_sl_keys if k in params}
        st.json(tp, expanded=True)
else:
    st.info("No hay parámetros registrados para este par.")
