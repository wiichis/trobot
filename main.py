import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import json
import atexit
import signal
import pkg.price_bingx_5m
import pkg.paper_momentum
import pkg.btc_bots
import pkg.btc_alertas
import pkg.funding_monitor
import pkg.light_mode
import pkg.daily_digest
import schedule
import time
import threading
import pkg
from pkg.lifecycle_events import emit_lifecycle_event


_BOT_STOP_EVENT_SENT = False

# Grupos de jobs que se registran al arrancar (pkg/jobs.json). Estos defaults son el estado
# vigente: si el archivo falta o no se puede leer, el bot de 5 min NO revive solo.
JOBS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pkg", "jobs.json")
JOBS_DEFAULT = {
    "bot_5m": False,
    "btc_bots": False,
    "paper_momentum": False,
    "funding_monitor": False,
    "btc_alertas": True,
    "daily_digest": True,
}


def _jobs_activos() -> dict:
    activos = dict(JOBS_DEFAULT)
    try:
        with open(JOBS_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k, v in cfg.items():
            if not k.startswith("_"):
                activos[k] = bool(v)
    except Exception as exc:
        print(f"⚠️ No pude leer {JOBS_PATH} ({exc}); uso los defaults de main.py")
    return activos
RUNTIME_CRITICAL_FILES = (
    "./archivos/tp_stage_state.csv",
    "./archivos/execution_ledger.csv",
    "./archivos/lifecycle_event_log.csv",
    "./archivos/order_submit_log.csv",
    "./archivos/order_lifecycle_log.csv",
)


def _runtime_storage_preflight() -> list[str]:
    issues: list[str] = []
    for rel_path in RUNTIME_CRITICAL_FILES:
        abs_path = os.path.abspath(rel_path)
        parent = os.path.dirname(abs_path)
        try:
            os.makedirs(parent, exist_ok=True)
        except Exception as exc:
            issues.append(f"{rel_path}:parent_mkdir_failed:{exc}")
            continue

        if os.path.exists(abs_path):
            if not os.access(abs_path, os.W_OK):
                issues.append(f"{rel_path}:not_writable")
            continue

        if not os.access(parent, os.W_OK):
            issues.append(f"{rel_path}:parent_not_writable")
            continue

        try:
            with open(abs_path, "a", encoding="utf-8"):
                pass
        except Exception as exc:
            issues.append(f"{rel_path}:touch_failed:{exc}")
    return issues


def _emit_bot_started(jobs: dict) -> None:
    symbols = []
    if jobs.get("bot_5m"):
        try:
            symbols = pkg.price_bingx_5m.currencies_list()
        except Exception:
            symbols = []
    emit_lifecycle_event(
        "bot_started",
        "INFO",
        jobs=",".join(k for k, v in jobs.items() if v),
        symbols_count=len(symbols) if symbols else None,
        symbols=",".join(symbols) if symbols else None,
    )


def _emit_bot_stopped(reason: str, severity: str = "WARN", force: bool = False) -> None:
    global _BOT_STOP_EVENT_SENT
    if _BOT_STOP_EVENT_SENT:
        return
    _BOT_STOP_EVENT_SENT = True
    emit_lifecycle_event(
        "bot_stopped",
        severity,
        force=force,
        reason=reason,
    )


def _handle_shutdown_signal(sig_num, _frame):
    _emit_bot_stopped(reason=f"signal_{sig_num}", severity="WARN", force=True)
    raise SystemExit(0)

def monkey_result():
    try:
        balance_actual, diferencia_hora, diferencia_dia, diferencia_semana = pkg.monkey_bx.monkey_result()

        def _fmt_line(label, val):
            signo = "+" if val >= 0 else ""
            emoji = "🟢" if val >= 0 else "🔴"
            num = f"{signo}{val:.2f}"
            return f"{label}  {emoji}`{num:>8}`"

        monkey_USD = "\n".join([
            f"{'━' * 14}",
            "📊 *RENDIMIENTO*",
            f"{'━' * 14}",
            f"💰 `{balance_actual:.2f} USD`",
            "",
            _fmt_line("1 hora", diferencia_hora),
            _fmt_line("hoy", diferencia_dia),
            _fmt_line("7 días", diferencia_semana),
        ])
        pkg.monkey_bx.bot_send_text(monkey_USD)
    except Exception as exc:
        print(f"⚠️ Error en monkey_result: {exc}")
        try:
            pkg.monkey_bx.bot_send_text(f"🚨 *Error en reporte de resultados*\n`{exc}`")
        except Exception:
            pass
    
def run_bingx():
    # Modo ligero (pkg/light_mode.py): sólo omite el recálculo de indicadores cuando el bot
    # está ocioso. El registro de órdenes y sus transiciones (colocando_ordenes) corre siempre.
    if not pkg.light_mode.skip_indicators():
        pkg.indicadores.update_indicators()
    pkg.monkey_bx.colocando_ordenes()

    
def run_fast():
    path = './archivos/indicadores.csv'
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        print("❌ indicadores.csv no existe o está vacío. Se omite run_fast.")
        return
    pkg.monkey_bx.colocando_TK_SL()
    
def posiciones_antiguas():
    pkg.monkey_bx.unrealized_profit_positions()



if __name__ == '__main__':
    jobs = _jobs_activos()
    print(f"Jobs activos: {', '.join(k for k, v in jobs.items() if v) or 'ninguno'}")
    if jobs["bot_5m"]:
        preflight_issues = _runtime_storage_preflight()
        if preflight_issues:
            detail = "; ".join(preflight_issues)
            print(f"⚠️ Runtime storage preflight con fallas: {detail}")
            emit_lifecycle_event(
                "runtime_storage_warning",
                "CRITICAL",
                source="startup_preflight",
                reason="runtime_storage_not_writable",
                detail=detail[:500],
            )
    _emit_bot_started(jobs)
    atexit.register(lambda: _emit_bot_stopped(reason="process_exit", severity="INFO"))
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    if jobs["bot_5m"]:
        # Bot de 5 min multi-par (perpetuo). Archivado el 22/10/2026: ver docs/legado/.
        for minute in range(1, 60, 5):
            schedule.every().hour.at(f":{minute:02d}").do(pkg.price_bingx_5m.price_bingx_5m)

        schedule.every(12).hours.at(":01").do(pkg.price_bingx_5m.actualizar_long_ultimas_12h)
        schedule.every().hour.at(":02").do(pkg.price_bingx_5m.completar_huecos_5m)
        schedule.every(12).hours.at(":02").do(pkg.price_bingx_5m.completar_ultimos_3dias)
        # Velas de los pares en BANCA (pkg/bench.json): no operan, sólo se simulan. El
        # minuto :04 no choca con el pull (:01,:06,...) ni con las entradas (:03,:08,...).
        schedule.every().hour.at(":04").do(pkg.price_bingx_5m.sync_bench_candles)

        # Colocar órdenes 2 minutos después de cada cierre de vela 5‑min (minutos 03, 08, ..., 58)
        for minute in range(3, 60, 5):
            schedule.every().hour.at(f":{minute:02d}").do(run_bingx)

        schedule.every(50).seconds.do(run_fast)
        schedule.every(6).hours.do(pkg.monkey_bx.resultado_PnL)
        schedule.every(5).minutes.do(posiciones_antiguas)
        # Reporte de resultados cada hora al minuto 59, ejecutado en un hilo independiente
        schedule.every().hour.at(":59").do(lambda: threading.Thread(target=monkey_result).start())
        # Resumen diario a las 23:00 UTC
        schedule.every().day.at("23:00").do(lambda: threading.Thread(target=pkg.monkey_bx.daily_summary).start())
        # Posiciones abiertas cada 6h
        schedule.every(6).hours.at(":30").do(lambda: threading.Thread(target=pkg.monkey_bx.open_positions_alert).start())

    if jobs["paper_momentum"]:
        # Prueba SIN DINERO del momentum entre pares (pkg/paper_momentum.json). Chequea cada
        # hora contra su estado en disco; rebalancea cada 72 h. No manda órdenes.
        schedule.every().hour.at(":05").do(pkg.paper_momentum.run_paper_momentum)
    if jobs["btc_alertas"]:
        # Alerta de ventana de compra/venta de BTC (pkg/btc_alertas.py): avisa cuando cambia.
        # Va antes que los bots para que su resumen diario lea el estado del día.
        schedule.every().hour.at(":02").do(pkg.btc_alertas.run_btc_alertas)
    if jobs["btc_bots"]:
        # Cuatro bots sobre BTC con libros virtuales (pkg/btc_bots.json). Cada hora; cada bot
        # decide una vez por vela cerrada (diaria o de 4 h). En modo papel no manda órdenes.
        schedule.every().hour.at(":02").do(pkg.btc_bots.run_btc_bots)
    if jobs["funding_monitor"]:
        # Monitor del carry de funding (pkg/funding_monitor.py): no opera, avisa si el funding
        # de los mejores pares entra al régimen alto. Mide una vez por día UTC.
        schedule.every().hour.at(":07").do(pkg.funding_monitor.run_funding_monitor_async)
    if jobs["daily_digest"]:
        # Resumen diario de BTC y bolsa de EE. UU. (pkg/daily_digest.py): un mensaje por día UTC
        # a partir de hora_utc (pkg/daily_digest.json). Hilo aparte; si una fuente cae, reintenta.
        schedule.every().hour.at(":12").do(pkg.daily_digest.run_daily_digest_async)

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as exc:
        _emit_bot_stopped(reason=f"runtime_exception:{str(exc)[:200]}", severity="CRITICAL", force=True)
        raise
