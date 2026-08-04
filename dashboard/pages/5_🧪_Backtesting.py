"""Página de Backtesting — Evaluación semanal de pares.

Lanza `scripts/evaluate_pairs.py` como subprocess detached, muestra log en
vivo, diff de params antes/después y permite desplegar con doble confirmación.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard import jobs, loaders

st.set_page_config(page_title="Backtesting — TRobot", page_icon="🧪", layout="wide")
st.title("🧪 Backtesting — Evaluación semanal")
st.caption("Lanza el flujo completo de `scripts/evaluate_pairs.py`: sync → clasifica → re-optimiza → rota → (deploy).")

JOB_KIND = "evaluate_pairs"
REPORT_PATH = _REPO_ROOT / "archivos" / "backtesting" / "evaluate_pairs" / "evaluation_report.json"
BEST_PROD_PATH = _REPO_ROOT / "pkg" / "best_prod.json"
REFRESH_SEC = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot_best_prod() -> Dict[str, Dict[str, Any]]:
    if not BEST_PROD_PATH.exists():
        return {}
    try:
        data = json.loads(BEST_PROD_PATH.read_text(encoding="utf-8"))
        return {e["symbol"]: e.get("params", {}) for e in data if isinstance(e, dict)}
    except Exception:
        return {}


def _load_report() -> Optional[Dict[str, Any]]:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _diff_params(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    rows = []
    for k in keys:
        b = before.get(k, "—")
        a = after.get(k, "—")
        if b != a:
            rows.append({"param": k, "antes": b, "después": a})
    return rows


def _fmt_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


# ---------------------------------------------------------------------------
# Sidebar: formulario de configuración
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Configuración")

with st.sidebar.form("bt_config"):
    st.caption("Ventanas de análisis")
    short_days = st.number_input("Ventana KEEP/EVALUATE (días)", 7, 30, 14, 1)
    long_days = st.number_input("Ventana peor performer (días)", 30, 180, 90, 5)
    opt_days = st.number_input("Ventana optimización (días)", 30, 180, 90, 5)

    st.caption("Sweep")
    n_trials = st.number_input("Random trials por símbolo", 50, 1000, 200, 50)
    max_attempts = st.number_input("Intentos candidato reemplazo", 1, 5, 3, 1)

    st.caption("Flujo")
    skip_sync = st.checkbox("Saltar sync desde producción", value=False,
                             help="Usa datos locales ya descargados")
    skip_rotation = st.checkbox("Saltar rotación (sin eliminar/reemplazar)", value=False)
    send_alert = st.checkbox("Enviar resumen por Telegram", value=True)

    st.markdown("---")
    st.caption("🔒 Acciones críticas (off por defecto)")
    apply_flag = st.checkbox("--apply · escribir a best_prod.json", value=False)
    deploy_flag = st.checkbox("--deploy · commit + push + merge + restart", value=False,
                               help="Requiere --apply")

    submit = st.form_submit_button("▶️ Lanzar evaluación", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Estado actual del job
# ---------------------------------------------------------------------------

active_id = jobs.is_locked()
current_status = jobs.get_status(active_id) if active_id else None

# ---------------------------------------------------------------------------
# Manejo del submit
# ---------------------------------------------------------------------------

if submit:
    if current_status and current_status["running"]:
        st.sidebar.error("⛔ Ya hay un job corriendo.")
    elif deploy_flag and not apply_flag:
        st.sidebar.error("--deploy requiere --apply.")
    elif deploy_flag:
        # Doble confirmación vía session_state
        st.session_state["_bt_pending_deploy"] = {
            "short_days": short_days, "long_days": long_days, "opt_days": opt_days,
            "n_trials": n_trials, "max_attempts": max_attempts,
            "skip_sync": skip_sync, "skip_rotation": skip_rotation,
            "send_alert": send_alert, "apply_flag": apply_flag, "deploy_flag": deploy_flag,
        }
    else:
        # Arranque inmediato
        cmd = [
            sys.executable, "-u", "scripts/evaluate_pairs.py",
            "--short_lookback_days", str(short_days),
            "--long_lookback_days", str(long_days),
            "--opt_lookback_days", str(opt_days),
            "--n_trials", str(n_trials),
            "--max_replacement_attempts", str(max_attempts),
        ]
        if skip_sync: cmd.append("--skip_sync")
        if skip_rotation: cmd.append("--skip_rotation")
        if send_alert: cmd.append("--send_alert")
        if apply_flag: cmd.append("--apply")

        snap = _snapshot_best_prod() if apply_flag else {}
        try:
            job = jobs.start_job(cmd, kind=JOB_KIND, meta={
                "apply": apply_flag, "deploy": False,
                "best_prod_snapshot": snap,
            })
            st.sidebar.success(f"Job lanzado: {job.id}")
            st.rerun()
        except RuntimeError as exc:
            st.sidebar.error(str(exc))


# ---------------------------------------------------------------------------
# Modal de confirmación para deploy
# ---------------------------------------------------------------------------

pending = st.session_state.get("_bt_pending_deploy")
if pending:
    st.error("⚠️ DEPLOY A PRODUCCIÓN")
    st.markdown(
        "Estás a punto de ejecutar con **`--apply --deploy`**. Esto:\n"
        "1. Re-optimiza pares con PnL negativo (sweep ~30-60 min).\n"
        "2. Escribe `pkg/best_prod.json` (con backup).\n"
        "3. Commit + push a la rama actual.\n"
        "4. SSH al servidor, `git merge`, `systemctl restart trobot`.\n\n"
        "**El bot se reiniciará.**"
    )
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("✅ Sí, desplegar ahora", type="primary"):
        cmd = [
            sys.executable, "-u", "scripts/evaluate_pairs.py",
            "--short_lookback_days", str(pending["short_days"]),
            "--long_lookback_days", str(pending["long_days"]),
            "--opt_lookback_days", str(pending["opt_days"]),
            "--n_trials", str(pending["n_trials"]),
            "--max_replacement_attempts", str(pending["max_attempts"]),
            "--apply", "--deploy",
        ]
        if pending["skip_sync"]: cmd.append("--skip_sync")
        if pending["skip_rotation"]: cmd.append("--skip_rotation")
        if pending["send_alert"]: cmd.append("--send_alert")

        snap = _snapshot_best_prod()
        try:
            job = jobs.start_job(cmd, kind=JOB_KIND, meta={
                "apply": True, "deploy": True,
                "best_prod_snapshot": snap,
            })
            st.session_state.pop("_bt_pending_deploy", None)
            st.success(f"Deploy iniciado: {job.id}")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    if c2.button("❌ Cancelar"):
        st.session_state.pop("_bt_pending_deploy", None)
        st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# Vista: job en curso
# ---------------------------------------------------------------------------

if current_status and current_status["running"]:
    st.subheader("🏃 Job en ejecución")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ID", current_status["id"])
    c2.metric("Duración", _fmt_duration(current_status["duration_sec"]))
    meta = current_status.get("meta", {})
    c3.metric("Apply", "✅" if meta.get("apply") else "—")
    c4.metric("Deploy", "✅" if meta.get("deploy") else "—")

    if current_status["duration_sec"] > 5400:  # 90 min
        st.warning("⏰ Job lleva más de 90 min. Considera detenerlo si está colgado.")

    cstop1, cstop2 = st.columns([1, 5])
    if cstop1.button("🛑 Detener job", type="secondary"):
        if jobs.stop_job(current_status["id"]):
            st.warning("Job detenido.")
            time.sleep(1)
            st.rerun()

    st.markdown("**Log en vivo** (auto-refresh cada 5s)")
    tail = jobs.read_log_tail(current_status["id"], n=300)
    st.code(tail or "(sin salida aún)", language="text")

    # Auto-refresh
    time.sleep(REFRESH_SEC)
    st.rerun()


# ---------------------------------------------------------------------------
# Vista: último job terminado (con diff de params)
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📋 Último resultado")

recent = jobs.list_recent_jobs(n=15, kind=JOB_KIND)
finished = [j for j in recent if j.get("finished_at")]

if not finished:
    st.info("Aún no se ha ejecutado ningún job. Configura y lanza desde el sidebar.")
else:
    last = finished[0]
    status = jobs.get_status(last["id"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ID", last["id"])
    c2.metric("Duración", _fmt_duration(status["duration_sec"]))
    exit_code = last.get("exit_code")
    if last.get("stopped_by_user"):
        c3.metric("Resultado", "🛑 detenido")
    elif exit_code == 0:
        c3.metric("Resultado", "✅ OK")
    else:
        c3.metric("Resultado", f"❌ exit {exit_code}")
    c4.metric("Inicio", last["started_at"].replace("T", " ").replace("Z", ""))

    report = _load_report()
    if report:
        # Keep
        keep = report.get("keep_pairs") or []
        if keep:
            st.markdown(f"**✅ Pares KEEP ({len(keep)})** — PnL positivo últimos {report.get('args', {}).get('short_lookback_days', 14)}d")
            df_keep = pd.DataFrame(keep)
            if not df_keep.empty and "pnl_14d" in df_keep.columns:
                df_keep["pnl_14d"] = df_keep["pnl_14d"].round(3)
            st.dataframe(df_keep, use_container_width=True, hide_index=True)

        # Evaluated (con diff de params)
        evaluated = report.get("evaluated_pairs") or []
        if evaluated:
            st.markdown(f"**🔧 Pares EVALUADOS ({len(evaluated)})**")
            snap = (last.get("meta") or {}).get("best_prod_snapshot", {})
            current_snap = _snapshot_best_prod()
            for ev in evaluated:
                sym = ev["symbol"]
                status_ev = ev.get("status", "?")
                pnl = ev.get("pnl_14d", 0.0)
                icon = {"REOPTIMIZED": "🔄", "DISABLED": "🚫"}.get(status_ev, "❓")
                with st.expander(f"{icon} {sym} — {status_ev} (PnL 14d: {pnl:+.2f})"):
                    if status_ev == "REOPTIMIZED":
                        before = snap.get(sym, {})
                        after = current_snap.get(sym, {})
                        diff = _diff_params(before, after)
                        if diff:
                            st.dataframe(pd.DataFrame(diff), use_container_width=True, hide_index=True)
                        else:
                            st.caption("(sin cambios detectados o snapshot no disponible)")
                    else:
                        st.caption(f"Estado: {status_ev}")

        # Rotación
        col_rem, col_rep = st.columns(2)
        rem = report.get("removed_pair")
        with col_rem:
            st.markdown("**🗑️ Eliminado**")
            if rem:
                st.write(f"**{rem['symbol']}** · PnL 90d: {rem.get('pnl_90d', 0):+.2f}")
            else:
                st.caption("—")
        rep = report.get("replacement_pair")
        with col_rep:
            st.markdown("**➕ Nuevo**")
            if rep:
                st.write(f"**{rep.get('symbol', '?')}**")
                st.json(rep, expanded=False)
            else:
                st.caption("—")

        # Estado final
        st.markdown("---")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Pares finales", report.get("final_pair_count", "?"))
        cc2.metric("Applied", "✅" if report.get("applied") else "—")
        cc3.metric("Deployed", "✅" if report.get("deployed") else "—")
    else:
        st.caption("(reporte JSON no encontrado)")

    with st.expander("📄 Log completo"):
        st.code(jobs.read_log_tail(last["id"], n=2000) or "(vacío)", language="text")


# ---------------------------------------------------------------------------
# Historial
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📚 Historial de ejecuciones")

if not recent:
    st.caption("Sin ejecuciones previas.")
else:
    rows = []
    for j in recent:
        st_j = jobs.get_status(j["id"]) or {}
        meta = j.get("meta") or {}
        result = "🏃 corriendo" if st_j.get("running") else (
            "🛑 detenido" if j.get("stopped_by_user")
            else ("✅ OK" if j.get("exit_code") == 0 else f"❌ exit {j.get('exit_code')}")
        )
        rows.append({
            "ID": j["id"],
            "Inicio": j["started_at"].replace("T", " ").replace("Z", ""),
            "Duración": _fmt_duration(st_j.get("duration_sec", 0)),
            "Apply": "✓" if meta.get("apply") else "",
            "Deploy": "✓" if meta.get("deploy") else "",
            "Resultado": result,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
