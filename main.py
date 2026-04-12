import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import os
import atexit
import signal
import pkg.price_bingx_5m
import schedule
import time
import threading
import pkg
from pkg.lifecycle_events import emit_lifecycle_event


_BOT_STOP_EVENT_SENT = False
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


def _emit_bot_started() -> None:
    symbols = []
    try:
        symbols = pkg.price_bingx_5m.currencies_list()
    except Exception:
        symbols = []
    emit_lifecycle_event(
        "bot_started",
        "INFO",
        symbols_count=len(symbols),
        symbols=",".join(symbols),
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

        def _fmt_delta(val):
            signo = "+" if val >= 0 else ""
            emoji = "🟢" if val >= 0 else "🔴"
            return f"{emoji} `{signo}{round(val, 2)} USD`"

        monkey_USD = "\n".join([
            f"{'━' * 20}",
            f"📊 *RESUMEN DE RENDIMIENTO*",
            f"{'━' * 20}",
            f"💰 *Balance actual:* `{round(balance_actual, 2)} USD`",
            "",
            f"▸ *Última hora:* {_fmt_delta(diferencia_hora)}",
            f"▸ *Hoy:* {_fmt_delta(diferencia_dia)}",
            f"▸ *Últimos 7 días:* {_fmt_delta(diferencia_semana)}",
        ])
        pkg.monkey_bx.bot_send_text(monkey_USD)
    except Exception as exc:
        print(f"⚠️ Error en monkey_result: {exc}")
        try:
            pkg.monkey_bx.bot_send_text(f"🚨 *Error en reporte de resultados*\n`{exc}`")
        except Exception:
            pass
    
def run_bingx():
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
    _emit_bot_started()
    atexit.register(lambda: _emit_bot_stopped(reason="process_exit", severity="INFO"))
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    for minute in range(1, 60, 5):
        schedule.every().hour.at(f":{minute:02d}").do(pkg.price_bingx_5m.price_bingx_5m)

    schedule.every(12).hours.at(":01").do(pkg.price_bingx_5m.actualizar_long_ultimas_12h)
    schedule.every().hour.at(":02").do(pkg.price_bingx_5m.completar_huecos_5m)
    schedule.every(12).hours.at(":02").do(pkg.price_bingx_5m.completar_ultimos_3dias)

    # Colocar órdenes 2 minutos después de cada cierre de vela 5‑min (ahora en minutos 03, 08, ..., 58)
    for minute in range(3, 60, 5):
        schedule.every().hour.at(f":{minute:02d}").do(run_bingx)

    schedule.every(50).seconds.do(run_fast)
    schedule.every(6).hours.do(pkg.monkey_bx.resultado_PnL)
    schedule.every(5).minutes.do(posiciones_antiguas)
    # Reporte de resultados cada hora al minuto 59, ejecutado en un hilo independiente
    schedule.every().hour.at(":59").do(lambda: threading.Thread(target=monkey_result).start())

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as exc:
        _emit_bot_stopped(reason=f"runtime_exception:{str(exc)[:200]}", severity="CRITICAL", force=True)
        raise
