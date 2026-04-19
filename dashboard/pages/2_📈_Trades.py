"""Página de histórico de trades."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import loaders

st.set_page_config(page_title="Trades — TRobot", page_icon="📈", layout="wide")

st.title("📈 Histórico de Trades")

pnl = loaders.load_pnl()
if pnl.empty:
    st.warning("No hay PnL.csv local. Ejecuta un sync desde la página principal.")
    st.stop()

realized = pnl[pnl["incomeType"] == "REALIZED_PNL"].copy()
if realized.empty:
    st.info("No hay trades REALIZED_PNL todavía.")
    st.stop()

realized["date"] = realized["time"].dt.date
realized["win"] = realized["income"] > 0

# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

c1, c2, c3 = st.columns(3)

symbols_all = ["(todos)"] + sorted(realized["symbol"].unique().tolist())
symbol_filter = c1.selectbox("Par", symbols_all, index=0)

min_date = realized["time"].min().date()
max_date = realized["time"].max().date()
date_range = c2.date_input(
    "Rango de fechas",
    value=(max(min_date, (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30)).date()), max_date),
    min_value=min_date, max_value=max_date,
)

result_filter = c3.selectbox("Resultado", ["(todos)", "Ganadores", "Perdedores"])

# ---------------------------------------------------------------------------
# Aplicar filtros
# ---------------------------------------------------------------------------

df = realized.copy()
if symbol_filter != "(todos)":
    df = df[df["symbol"] == symbol_filter]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start, end = date_range
    df = df[(df["date"] >= start) & (df["date"] <= end)]
if result_filter == "Ganadores":
    df = df[df["win"]]
elif result_filter == "Perdedores":
    df = df[~df["win"]]

st.markdown("---")

# ---------------------------------------------------------------------------
# KPIs filtrados
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", len(df))
c2.metric("PnL total", f"{df['income'].sum():+.2f}")
c3.metric("Winrate", f"{df['win'].mean()*100:.1f}%" if len(df) else "—")
c4.metric("Promedio", f"{df['income'].mean():+.3f}" if len(df) else "—")

st.markdown("---")

# ---------------------------------------------------------------------------
# Distribución
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

# Histograma de PnL
fig = px.histogram(
    df, x="income", nbins=30,
    color=df["win"].map({True: "Ganador", False: "Perdedor"}),
    color_discrete_map={"Ganador": "#2ecc71", "Perdedor": "#e74c3c"},
    title="Distribución de PnL por trade",
)
fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
col_a.plotly_chart(fig, use_container_width=True)

# Actividad por día de semana / hora
df["hour_utc"] = df["time"].dt.hour
df["dow"] = df["time"].dt.day_name()
heat = df.groupby(["dow", "hour_utc"])["income"].agg(["count", "sum"]).reset_index()

if not heat.empty:
    fig2 = px.density_heatmap(
        heat, x="hour_utc", y="dow", z="sum",
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        title="Heatmap: PnL por día × hora UTC",
        labels={"hour_utc": "Hora UTC", "dow": "Día", "sum": "PnL"},
    )
    fig2.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    col_b.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabla detallada
# ---------------------------------------------------------------------------

st.subheader("Detalle de trades")

show = df.copy().sort_values("time", ascending=False)
show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M")
show["income"] = show["income"].round(4)
show = show[["time", "symbol", "income", "asset"]].rename(columns={
    "time": "Fecha UTC", "symbol": "Par", "income": "PnL", "asset": "Asset",
})
st.dataframe(show, use_container_width=True, hide_index=True, height=400)
