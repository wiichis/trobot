"""TRobot Dashboard — página principal (Resumen).

Uso:
    # Primera vez:
    python3 -m venv .venv-dashboard
    source .venv-dashboard/bin/activate
    pip install -r dashboard/requirements.txt

    # Lanzar:
    streamlit run dashboard/app.py

    Abre http://localhost:8501
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import loaders, sync

st.set_page_config(
    page_title="TRobot Dashboard",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar: sync + info
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    st.sidebar.title("🤖 TRobot")
    st.sidebar.caption("Dashboard de rendimiento")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Sincronizar datos")

    col1, col2 = st.sidebar.columns([3, 1])
    if col1.button("Sync desde producción", use_container_width=True, type="primary"):
        with st.spinner("Descargando CSV del servidor..."):
            results = sync.sync_from_production()
        loaders.clear_all_caches()
        ok_count = sum(1 for r in results if r.ok and not r.skipped)
        skipped_count = sum(1 for r in results if r.skipped)
        failed = [r for r in results if not r.ok]
        total = len(results)
        if not failed:
            msg = f"✅ {ok_count}/{total} archivos sincronizados"
            if skipped_count:
                msg += f" ({skipped_count} opcionales no existen)"
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(f"⚠️ {ok_count}/{total} archivos OK")
            for r in failed:
                st.sidebar.caption(f"❌ {r.file_rel}: {r.detail[:80]}")
        st.session_state["last_sync"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if "last_sync" in st.session_state:
        st.sidebar.caption(f"Último sync: {st.session_state['last_sync']}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Rango de análisis")
    default_days = st.sidebar.selectbox(
        "Ventana",
        options=[1, 7, 14, 30, 90, 180],
        index=2,
        format_func=lambda d: f"Últimos {d} días",
        key="range_days",
    )
    return default_days


# ---------------------------------------------------------------------------
# KPIs cards
# ---------------------------------------------------------------------------

def render_kpi_row(pnl_df: pd.DataFrame, balance_df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    # Balance actual
    if not balance_df.empty:
        bal_now = balance_df["balance"].iloc[-1]
        if len(balance_df) > 1:
            bal_24h_ago = balance_df[balance_df["date"] <= (balance_df["date"].iloc[-1] - pd.Timedelta(hours=24))]
            delta_24h = bal_now - bal_24h_ago["balance"].iloc[-1] if not bal_24h_ago.empty else 0
        else:
            delta_24h = 0
        col1.metric("💰 Balance", f"{bal_now:.2f} USD", f"{delta_24h:+.2f} (24h)")
    else:
        col1.metric("💰 Balance", "—")

    # Ventanas de PnL
    s7 = loaders.pnl_summary(pnl_df, 7)
    s30 = loaders.pnl_summary(pnl_df, 30)

    col2.metric("📈 PnL 7d", f"{s7['pnl_net']:+.2f}", f"{s7['trades']} trades")
    col3.metric("📊 PnL 30d", f"{s30['pnl_net']:+.2f}", f"{s30['trades']} trades")
    col4.metric("🎯 Winrate 30d", f"{s30['winrate']:.1f}%")
    pf_disp = f"{s30['profit_factor']:.2f}" if s30["profit_factor"] < 99 else "∞"
    col5.metric("⚡ Profit Factor 30d", pf_disp)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def render_pnl_chart(pnl_df: pd.DataFrame, days: int) -> None:
    st.subheader(f"PnL acumulado — últimos {days} días")

    if pnl_df.empty:
        st.info("No hay datos de PnL.")
        return

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = pnl_df[pnl_df["time"] >= cutoff]
    daily = loaders.daily_pnl(df)

    if daily.empty:
        st.info("Sin trades en esta ventana.")
        return

    fig = go.Figure()
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in daily["pnl_day"]]
    fig.add_bar(x=daily["day"], y=daily["pnl_day"], name="PnL diario", marker_color=colors, opacity=0.5)
    fig.add_scatter(
        x=daily["day"], y=daily["pnl_cum"], mode="lines+markers",
        name="PnL acumulado", line=dict(color="#3498db", width=3), yaxis="y2",
    )
    fig.update_layout(
        height=380,
        yaxis=dict(title="PnL diario (USD)", side="left"),
        yaxis2=dict(title="Acumulado (USD)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_balance_chart(balance_df: pd.DataFrame, days: int) -> None:
    st.subheader(f"Evolución del balance — últimos {days} días")

    if balance_df.empty:
        st.info("No hay datos de balance (ganancias.csv).")
        return

    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)
    df = balance_df[balance_df["date"] >= cutoff]
    if df.empty:
        st.info("Sin snapshots en la ventana.")
        return

    fig = px.area(df, x="date", y="balance", title=None)
    fig.update_traces(line=dict(color="#3498db", width=2), fillcolor="rgba(52, 152, 219, 0.15)")
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="USD", xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_per_symbol(pnl_df: pd.DataFrame, days: int) -> None:
    st.subheader(f"Rendimiento por par — últimos {days} días")

    agg = loaders.trades_by_symbol(pnl_df, days=days)
    if agg.empty:
        st.info("Sin trades en la ventana.")
        return

    c1, c2 = st.columns([2, 1])

    # Barras de PnL neto por par
    fig = px.bar(
        agg, x="symbol", y="pnl_net",
        color=agg["pnl_net"].apply(lambda x: "Positivo" if x >= 0 else "Negativo"),
        color_discrete_map={"Positivo": "#2ecc71", "Negativo": "#e74c3c"},
        labels={"pnl_net": "PnL neto (USD)", "symbol": ""},
    )
    fig.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    c1.plotly_chart(fig, use_container_width=True)

    # Tabla
    show = agg[["symbol", "trades", "pnl_net", "winrate", "profit_factor"]].copy()
    show.columns = ["Par", "Trades", "PnL neto", "Winrate %", "PF"]
    show["PnL neto"] = show["PnL neto"].round(3)
    show["Winrate %"] = show["Winrate %"].round(1)
    show["PF"] = show["PF"].apply(lambda x: "∞" if x >= 99 else f"{x:.2f}")
    c2.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    days = render_sidebar()

    st.title("📊 Resumen de rendimiento")
    st.caption("Datos descargados vía SCP del servidor de producción. Presiona '🔄 Sync' en el sidebar para actualizar.")

    pnl = loaders.load_pnl()
    balance = loaders.load_ganancias()

    if pnl.empty and balance.empty:
        st.warning("No hay datos locales. Ejecuta un sync primero.")
        return

    render_kpi_row(pnl, balance)
    st.markdown("---")
    render_pnl_chart(pnl, days)
    render_balance_chart(balance, days)
    st.markdown("---")
    render_per_symbol(pnl, days)


if __name__ == "__main__":
    main()
