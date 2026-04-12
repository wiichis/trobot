#!/usr/bin/env python3
"""evaluate_pairs.py -- Evaluacion periodica y rotacion de pares para TRobot.

Flujo:
  1. Sincronizar PnL.csv y cripto_price_5m_long.csv desde el servidor de produccion (SCP).
  2. Clasificar pares por PnL real (14d): KEEP (positivo) o EVALUATE (negativo).
  3. Re-optimizar pares EVALUATE con backtesting sweep.
  4. Identificar el peor performer de 90d para eliminacion.
  5. Buscar reemplazo entre los pares de mayor volumen de BingX.
  6. Actualizar best_prod.json y enviar resumen por Telegram.
  7. (Opcional) Desplegar a produccion y reiniciar el bot.

Uso tipico:
  python3 scripts/evaluate_pairs.py --send_alert            # dry run
  python3 scripts/evaluate_pairs.py --apply --send_alert     # aplicar local
  python3 scripts/evaluate_pairs.py --apply --deploy --send_alert  # full deploy
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg import backtesting  # noqa: E402

DEFAULT_PEM_KEY = Path.home() / "Documents" / "proyectos" / "ls_keys" / "trobot4.pem"
DEFAULT_SERVER = "ubuntu@98.81.217.194"
REMOTE_BASE = "/home/ubuntu/TRobot"

# Archivos de datos que se sincronizan del servidor
SYNC_FILES = [
    ("archivos/PnL.csv", f"{REMOTE_BASE}/archivos/PnL.csv"),
    ("archivos/cripto_price_5m_long.csv", f"{REMOTE_BASE}/archivos/cripto_price_5m_long.csv"),
]

# BingX public endpoints
BINGX_CONTRACTS_URL = "https://open-api.bingx.com/openApi/swap/v2/quote/contracts"
BINGX_TICKER_URL = "https://open-api.bingx.com/openApi/swap/v2/quote/ticker"


# ===========================================================================
# Helpers generales
# ===========================================================================

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ts_label() -> str:
    return _now_utc().strftime("%Y%m%d_%H%M%S")


def _load_best_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Best file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Invalid best file format (expected list): {path}")
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).upper().strip()
        params = item.get("params", {})
        if not symbol or not isinstance(params, dict):
            continue
        out.append({"symbol": symbol, "params": params})
    return out


def _entries_to_map(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(e["symbol"]).upper(): e for e in entries}


def _save_best_entries(path: Path, entries: List[Dict[str, Any]], backup: bool = True) -> Optional[Path]:
    """Guarda entries como JSON. Si backup=True, crea respaldo timestamped."""
    bak_path = None
    if backup and path.exists():
        bak_path = path.with_suffix(f".json.bak.{_ts_label()}")
        shutil.copy2(path, bak_path)
        print(f"  Backup: {bak_path}")
    sorted_entries = sorted(entries, key=lambda e: e.get("symbol", ""))
    path.write_text(json.dumps(sorted_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  Escrito: {path} ({len(entries)} pares)")
    return bak_path


# ===========================================================================
# Fase 1: Sincronizar datos desde produccion
# ===========================================================================

def _sync_from_production(server: str, pem_key: Path) -> bool:
    """Descarga PnL.csv y cripto_price_5m_long.csv del servidor via SCP."""
    if not pem_key.exists():
        print(f"  ERROR: PEM key no encontrada: {pem_key}")
        return False

    ok = True
    for local_rel, remote_path in SYNC_FILES:
        local_path = REPO_ROOT / local_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "scp", "-i", str(pem_key),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            f"{server}:{remote_path}",
            str(local_path),
        ]
        print(f"  SCP: {remote_path} -> {local_rel}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"    OK ({size_mb:.1f} MB)")
        except subprocess.CalledProcessError as exc:
            print(f"    FALLO: {exc.stderr.decode()[:200] if exc.stderr else exc}")
            ok = False
        except subprocess.TimeoutExpired:
            print("    FALLO: timeout (120s)")
            ok = False
    return ok


# ===========================================================================
# Fase 2: Clasificar pares por PnL real
# ===========================================================================

def _compute_pnl_by_symbol(
    pnl_csv: Path,
    lookback_days: int,
    income_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Lee PnL.csv y agrupa por symbol. Devuelve DataFrame [symbol, pnl_net, trade_count]."""
    if not pnl_csv.exists():
        raise FileNotFoundError(f"PnL file not found: {pnl_csv}")
    df = pd.read_csv(pnl_csv)
    required = {"symbol", "income", "incomeType", "time"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"PnL file missing columns: {missing}")

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["income"] = pd.to_numeric(df["income"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["symbol", "income", "time"])

    if income_types:
        allow = {x.strip().upper() for x in income_types if x.strip()}
        df = df[df["incomeType"].astype(str).str.upper().isin(allow)]

    if df.empty:
        return pd.DataFrame(columns=["symbol", "pnl_net", "trade_count"])

    end_ts = df["time"].max()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    recent = df[df["time"] >= start_ts].copy()

    if recent.empty:
        return pd.DataFrame(columns=["symbol", "pnl_net", "trade_count"])

    grouped = (
        recent.groupby("symbol", as_index=False)
        .agg(pnl_net=("income", "sum"), trade_count=("income", "size"))
        .sort_values("pnl_net", ascending=True)
        .reset_index(drop=True)
    )
    return grouped


def _classify_pairs(
    pnl_df: pd.DataFrame,
    active_symbols: List[str],
) -> Tuple[List[str], List[str]]:
    """Clasifica en KEEP (PnL > 0) y EVALUATE (PnL <= 0)."""
    pnl_map = {}
    for _, row in pnl_df.iterrows():
        pnl_map[str(row["symbol"]).upper()] = float(row["pnl_net"])

    keep, evaluate = [], []
    for sym in active_symbols:
        pnl = pnl_map.get(sym.upper(), 0.0)
        if pnl > 0:
            keep.append(sym)
        else:
            evaluate.append(sym)
    return keep, evaluate


# ===========================================================================
# Fase 3: Re-optimizar pares EVALUATE
# ===========================================================================

def _run_sweep_for_symbols(
    symbols: List[str],
    data_template: str,
    sweep_cfg: str,
    out_dir: Path,
    lookback_days: int,
    n_trials: int,
    candidate_best_path: Path,
) -> None:
    """Corre backtesting sweep via subprocess."""
    cmd = [
        sys.executable,
        "pkg/backtesting.py",
        "--symbols", ",".join(symbols),
        "--data_template", data_template,
        "--lookback_days", str(int(lookback_days)),
        "--train_ratio", "0",
        "--search_mode", "random",
        "--n_trials", str(int(n_trials)),
        "--rank_by", "pnl_net",
        "--sweep", sweep_cfg,
        "--out_dir", str(out_dir),
        "--min_trades", "5",
        "--max_cost_ratio", "1.0",
        "--export_best", str(candidate_best_path),
        "--export_positive_ratio",
    ]
    print(f"  Sweep: {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _eval_symbol(
    best_path: Path, symbol: str, data_template: str, capital: float, days: int,
) -> Dict[str, Any]:
    """Evalua un simbolo con live_parity_portfolio."""
    res = backtesting.run_live_parity_portfolio(
        symbols=[symbol],
        data_template=data_template,
        capital=float(capital),
        weights_csv=None,
        best_path=str(best_path),
        lookback_days=int(days),
    )
    return {
        "trades": int(res.get("trades", 0) or 0),
        "pnl_net": float(res.get("pnl_net", 0.0) or 0.0),
        "cost_ratio": None if res.get("cost_ratio") is None else float(res["cost_ratio"]),
        "max_dd_pct": None if res.get("max_dd_pct") is None else float(res["max_dd_pct"]),
        "winrate_pct": float(res.get("winrate_pct", 0.0) or 0.0),
    }


# ===========================================================================
# Fase 5: Buscar reemplazo por volumen
# ===========================================================================

def _fetch_top_volume_pairs(exclude: set, max_candidates: int = 10) -> List[str]:
    """Consulta BingX ticker API para obtener pares perpetuos ordenados por volumen 24h.
    Filtra pares exoticos y solo devuelve pares 'limpios' (3-5 letras base)."""
    exclude_upper = {s.upper() for s in exclude}
    pairs: List[Tuple[str, float]] = []

    # Usar endpoint ticker (tiene quoteVolume)
    try:
        resp = requests.get(BINGX_TICKER_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if isinstance(data, list):
            for item in data:
                sym = str(item.get("symbol", "")).strip().upper()
                if not sym or not sym.endswith("-USDT"):
                    continue
                # Filtrar pares exoticos: solo base de 2-6 letras alfanumericas
                base = sym.replace("-USDT", "")
                if not base.isalpha() or len(base) < 2 or len(base) > 6:
                    continue
                vol = 0.0
                for key in ("quoteVolume", "volume"):
                    if key in item and item[key]:
                        try:
                            vol = float(item[key])
                            break
                        except (ValueError, TypeError):
                            continue
                if sym not in exclude_upper and vol > 0:
                    pairs.append((sym, vol))
    except Exception as exc:
        print(f"  WARN: ticker endpoint fallo: {exc}")

    # Ordenar por volumen desc, devolver top N
    pairs.sort(key=lambda x: x[1], reverse=True)
    result = [p[0] for p in pairs[:max_candidates]]
    if result:
        print(f"  Top {len(result)} por volumen 24h:")
        for i, (sym, vol) in enumerate(pairs[:max_candidates]):
            print(f"    {i+1}. {sym}: {vol/1e6:.1f}M USDT")
    return result


def _download_full_history_for_symbol(
    symbol: str,
    long_csv: Path,
    max_batches: int = 52,
) -> bool:
    """Descarga historial de velas 5m para un simbolo nuevo y lo agrega al CSV largo."""
    # Importar la funcion de descarga del modulo existente
    from pkg.price_bingx_5m import _fetch_bingx_candles  # noqa: E402

    print(f"  Descargando historial para {symbol} (max {max_batches} batches)...")
    all_candles: List[Dict] = []
    end_time_ms: Optional[int] = None

    for batch_n in range(1, max_batches + 1):
        try:
            candles = _fetch_bingx_candles(symbol, 1000, end_time_ms)
        except Exception as exc:
            print(f"    Batch {batch_n}: error {exc}")
            break
        if not candles:
            print(f"    Batch {batch_n}: sin datos, deteniendo")
            break
        all_candles.extend(candles)
        # Retroceder: usar la vela mas antigua - 5min
        oldest = min(c["date"] for c in candles)
        end_time_ms = int(oldest.timestamp() * 1000) - 300_000
        if batch_n % 10 == 0:
            print(f"    Batch {batch_n}/{max_batches}: {len(all_candles)} velas acumuladas")
        _time.sleep(0.3)

    if not all_candles:
        print(f"  No se obtuvieron velas para {symbol}")
        return False

    df_new = pd.DataFrame(all_candles)
    df_new["date"] = pd.to_datetime(df_new["date"], utc=True)
    print(f"  Descargadas {len(df_new)} velas para {symbol} "
          f"({df_new['date'].min()} -> {df_new['date'].max()})")

    # Leer CSV largo existente y concatenar
    if long_csv.exists() and long_csv.stat().st_size > 0:
        df_existing = pd.read_csv(long_csv)
        df_existing["date"] = pd.to_datetime(df_existing["date"], utc=True)
        df_concat = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_concat = df_new

    df_concat = df_concat.drop_duplicates(subset=["symbol", "date"], keep="last")
    df_concat = df_concat.sort_values(["symbol", "date"]).reset_index(drop=True)
    df_concat.to_csv(long_csv, index=False)
    print(f"  CSV largo actualizado: {len(df_concat)} filas totales")
    return True


# ===========================================================================
# Telegram
# ===========================================================================

def _send_telegram_if_possible(message: str) -> Tuple[bool, str]:
    try:
        import pkg.credentials as credentials
    except Exception as exc:
        return False, f"credentials import failed: {exc}"

    token = getattr(credentials, "token", None)
    chat_id = getattr(credentials, "chatID", None) or getattr(credentials, "chat_id", None)
    if not token or not chat_id:
        return False, "missing token/chat_id in pkg.credentials"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload, timeout=12)
    except Exception as exc:
        return False, f"telegram request failed: {exc}"
    if not resp.ok:
        body = resp.text[:220] if resp.text else ""
        return False, f"telegram status={resp.status_code} body={body}"
    return True, "ok"


def _format_summary(report: Dict[str, Any]) -> str:
    """Formatea el reporte como mensaje de Telegram."""
    lines = []
    lines.append(f"{'━' * 22}")
    lines.append("🤖 *TRobot | Evaluacion de Pares*")
    lines.append(f"{'━' * 22}")
    lines.append(f"🕒 {report.get('timestamp', '')}")
    lines.append("")

    # KEEP
    keep = report.get("keep_pairs", [])
    if keep:
        lines.append(f"✅ *KEEP* ({len(keep)} pares — PnL positivo 14d)")
        for p in keep:
            pnl = float(p.get("pnl_14d", 0))
            emoji = "🟢" if pnl > 0 else "⚪"
            sym = p["symbol"].replace("-USDT", "")
            lines.append(f"  {emoji} {sym}: `{pnl:+.2f} USD`")
        lines.append("")

    # EVALUATE
    evaluated = report.get("evaluated_pairs", [])
    if evaluated:
        lines.append(f"🔄 *RE-OPTIMIZADOS* ({len(evaluated)} pares)")
        for p in evaluated:
            sym = p["symbol"].replace("-USDT", "")
            status = p.get("status", "")
            pnl_14d = float(p.get("pnl_14d", 0))
            if status == "REOPTIMIZED":
                lines.append(f"  🛠 {sym}: PnL 14d=`{pnl_14d:+.2f}` → nuevos params")
            elif status == "DISABLED":
                lines.append(f"  ❌ {sym}: PnL 14d=`{pnl_14d:+.2f}` → desactivado")
            else:
                lines.append(f"  ⚠️ {sym}: PnL 14d=`{pnl_14d:+.2f}` → {status}")
        lines.append("")

    # Removed
    removed = report.get("removed_pair")
    if removed:
        sym = removed["symbol"].replace("-USDT", "")
        pnl_90d = float(removed.get("pnl_90d", 0))
        lines.append(f"🗑 *ELIMINADO* (peor 90d)")
        lines.append(f"  {sym}: PnL 90d=`{pnl_90d:+.2f} USD`")
        lines.append("")

    # Replacement
    replacement = report.get("replacement_pair")
    if replacement:
        sym = replacement["symbol"].replace("-USDT", "")
        bt_pnl = float(replacement.get("bt_pnl", 0))
        lines.append(f"🆕 *NUEVO PAR*")
        lines.append(f"  {sym}: BT PnL=`{bt_pnl:+.2f}` (top volumen BingX)")
        lines.append("")

    # Summary
    applied = report.get("applied", False)
    deployed = report.get("deployed", False)
    total = report.get("final_pair_count", "?")
    lines.append(f"📊 Pares finales: *{total}*")
    lines.append(f"💾 Aplicado: {'✅' if applied else '❌ (dry run)'}")
    if applied:
        lines.append(f"🚀 Desplegado: {'✅' if deployed else '❌ (pendiente)'}")

    msg = "\n".join(lines)
    if len(msg) > 3900:
        msg = msg[:3850] + "\n...(truncado)"
    return msg


# ===========================================================================
# Fase 7: Deploy
# ===========================================================================

def _deploy_to_production(server: str, pem_key: Path, branch: str) -> bool:
    """Commit, push, merge en produccion y reiniciar trobot."""
    ssh_base = [
        "ssh", "-i", str(pem_key),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        server,
    ]

    # 1. Commit local
    print("  [DEPLOY] Commit local...")
    try:
        subprocess.run(
            ["git", "add", "pkg/best_prod.json"],
            cwd=str(REPO_ROOT), check=True, capture_output=True,
        )
        msg = f"chore: actualizar best_prod.json via evaluate_pairs ({_ts_label()})"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(REPO_ROOT), check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode()[:200] if exc.stderr else str(exc)
        # Si no hay cambios para commit, no es error fatal
        if "nothing to commit" in stderr:
            print("    Nada que commitear (best_prod.json sin cambios en git)")
        else:
            print(f"    FALLO commit: {stderr}")
            return False

    # 2. Push
    print(f"  [DEPLOY] Push a origin/{branch}...")
    try:
        subprocess.run(
            ["git", "push", "origin", branch],
            cwd=str(REPO_ROOT), check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:
        print(f"    FALLO push: {exc}")
        return False

    # 3. Merge en servidor
    print("  [DEPLOY] Merge en produccion...")
    merge_cmd = f"cd {REMOTE_BASE} && git fetch origin && git merge origin/{branch} --no-edit"
    try:
        subprocess.run(
            ssh_base + [merge_cmd],
            check=True, capture_output=True, timeout=30,
        )
    except Exception as exc:
        print(f"    FALLO merge: {exc}")
        return False

    # 4. Restart
    print("  [DEPLOY] Reiniciando trobot...")
    try:
        subprocess.run(
            ssh_base + ["sudo systemctl restart trobot"],
            check=True, capture_output=True, timeout=15,
        )
        _time.sleep(2)
        result = subprocess.run(
            ssh_base + ["sudo systemctl is-active trobot"],
            capture_output=True, timeout=10,
        )
        status = result.stdout.decode().strip()
        if status == "active":
            print(f"    trobot: {status} ✅")
            return True
        else:
            print(f"    trobot: {status} ⚠️")
            return False
    except Exception as exc:
        print(f"    FALLO restart: {exc}")
        return False


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluacion periodica y rotacion de pares para TRobot.",
    )
    # Paths
    parser.add_argument("--pnl_csv", default="archivos/PnL.csv")
    parser.add_argument("--best_prod", default="pkg/best_prod.json")
    parser.add_argument("--data_template", default="archivos/cripto_price_5m_long.csv")
    parser.add_argument("--sweep_cfg", default="archivos/backtesting/simple_sweep.json")
    parser.add_argument("--out_dir", default="archivos/backtesting/evaluate_pairs")
    parser.add_argument("--report", default="archivos/backtesting/evaluate_pairs/evaluation_report.json")

    # Lookback windows
    parser.add_argument("--short_lookback_days", type=int, default=14,
                        help="Dias para clasificar KEEP/EVALUATE (default: 14)")
    parser.add_argument("--long_lookback_days", type=int, default=90,
                        help="Dias para identificar peor performer (default: 90)")
    parser.add_argument("--opt_lookback_days", type=int, default=90,
                        help="Dias de data para backtesting sweep (default: 90)")

    # Sweep params
    parser.add_argument("--n_trials", type=int, default=200,
                        help="Random search trials por simbolo (default: 200)")
    parser.add_argument("--max_replacement_attempts", type=int, default=3,
                        help="Intentos con candidatos nuevos (default: 3)")
    parser.add_argument("--max_history_batches", type=int, default=52,
                        help="Batches de API para historial de nuevo par (default: 52 ~ 180 dias)")

    # Server
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"SSH destino (default: {DEFAULT_SERVER})")
    parser.add_argument("--pem_key", default=str(DEFAULT_PEM_KEY),
                        help=f"Ruta PEM key (default: {DEFAULT_PEM_KEY})")

    # Flags
    parser.add_argument("--skip_sync", action="store_true",
                        help="Saltar descarga de datos del servidor")
    parser.add_argument("--skip_rotation", action="store_true",
                        help="Saltar eliminacion/reemplazo de pares")
    parser.add_argument("--apply", action="store_true",
                        help="Escribir cambios a best_prod.json")
    parser.add_argument("--deploy", action="store_true",
                        help="Commit, push, merge en produccion y reiniciar (requiere --apply)")
    parser.add_argument("--send_alert", action="store_true",
                        help="Enviar resumen por Telegram")

    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = REPO_ROOT / args.report
    best_prod_path = REPO_ROOT / args.best_prod
    pnl_path = REPO_ROOT / args.pnl_csv
    pem_key = Path(args.pem_key).expanduser()

    report: Dict[str, Any] = {
        "timestamp": _now_utc().isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    # ── Fase 1: Sync datos ──────────────────────────────────────────────
    print("\n═══ FASE 1: Sincronizar datos desde produccion ═══")
    if args.skip_sync:
        print("  (saltado: --skip_sync)")
        report["sync"] = "skipped"
    else:
        sync_ok = _sync_from_production(args.server, pem_key)
        report["sync"] = "ok" if sync_ok else "partial_failure"
        if not sync_ok:
            print("  WARN: Sincronizacion parcial. Se continua con datos disponibles.")

    # ── Cargar estado actual ────────────────────────────────────────────
    current_entries = _load_best_entries(best_prod_path)
    current_symbols = [e["symbol"] for e in current_entries]
    current_map = _entries_to_map(current_entries)
    print(f"\n  Pares actuales en best_prod.json: {len(current_symbols)}")
    for sym in current_symbols:
        print(f"    • {sym}")

    # ── Fase 2: Clasificar por PnL real ─────────────────────────────────
    print(f"\n═══ FASE 2: Clasificar pares (PnL {args.short_lookback_days}d) ═══")
    try:
        pnl_14d = _compute_pnl_by_symbol(
            pnl_path,
            lookback_days=args.short_lookback_days,
            income_types=["REALIZED_PNL"],
        )
    except Exception as exc:
        print(f"  ERROR leyendo PnL: {exc}")
        pnl_14d = pd.DataFrame(columns=["symbol", "pnl_net", "trade_count"])

    pnl_map_14d = {str(r["symbol"]).upper(): float(r["pnl_net"]) for _, r in pnl_14d.iterrows()}
    keep_symbols, evaluate_symbols = _classify_pairs(pnl_14d, current_symbols)

    print(f"\n  ✅ KEEP ({len(keep_symbols)}):")
    keep_report = []
    for sym in keep_symbols:
        pnl = pnl_map_14d.get(sym, 0.0)
        print(f"    {sym}: PnL 14d = {pnl:+.2f} USD")
        keep_report.append({"symbol": sym, "pnl_14d": pnl})

    print(f"\n  🔄 EVALUATE ({len(evaluate_symbols)}):")
    for sym in evaluate_symbols:
        pnl = pnl_map_14d.get(sym, 0.0)
        print(f"    {sym}: PnL 14d = {pnl:+.2f} USD")

    report["keep_pairs"] = keep_report

    # ── Fase 3: Re-optimizar pares EVALUATE ─────────────────────────────
    print(f"\n═══ FASE 3: Re-optimizar pares EVALUATE ═══")
    evaluated_report: List[Dict[str, Any]] = []
    new_params_map: Dict[str, Dict[str, Any]] = {}

    if not evaluate_symbols:
        print("  No hay pares para re-optimizar. Todos tienen PnL positivo.")
    else:
        candidate_best = out_dir / "best_prod_evaluate_candidate.json"
        try:
            _run_sweep_for_symbols(
                evaluate_symbols,
                args.data_template,
                args.sweep_cfg,
                out_dir,
                args.opt_lookback_days,
                args.n_trials,
                candidate_best,
            )

            # Leer resultados del sweep
            if candidate_best.exists():
                candidate_entries = _load_best_entries(candidate_best)
                candidate_map = _entries_to_map(candidate_entries)
            else:
                candidate_map = {}

            for sym in evaluate_symbols:
                pnl = pnl_map_14d.get(sym, 0.0)
                if sym in candidate_map:
                    new_entry = candidate_map[sym]
                    print(f"  ✅ {sym}: Re-optimizado con nuevos params")
                    new_params_map[sym] = new_entry
                    evaluated_report.append({
                        "symbol": sym, "pnl_14d": pnl, "status": "REOPTIMIZED",
                    })
                else:
                    print(f"  ❌ {sym}: No paso filtros de backtesting → DISABLED")
                    evaluated_report.append({
                        "symbol": sym, "pnl_14d": pnl, "status": "DISABLED",
                    })

        except Exception as exc:
            print(f"  ERROR en sweep: {exc}")
            for sym in evaluate_symbols:
                pnl = pnl_map_14d.get(sym, 0.0)
                evaluated_report.append({
                    "symbol": sym, "pnl_14d": pnl, "status": f"SWEEP_FAILED: {str(exc)[:100]}",
                })

    report["evaluated_pairs"] = evaluated_report

    # ── Fase 4: Identificar peor performer (90d) ───────────────────────
    print(f"\n═══ FASE 4: Identificar peor performer ({args.long_lookback_days}d) ═══")
    worst_symbol: Optional[str] = None
    worst_pnl_90d: float = 0.0

    if args.skip_rotation:
        print("  (saltado: --skip_rotation)")
    else:
        try:
            pnl_90d = _compute_pnl_by_symbol(
                pnl_path,
                lookback_days=args.long_lookback_days,
                income_types=["REALIZED_PNL", "TRADING_FEE", "FUNDING_FEE"],
            )
            pnl_map_90d = {str(r["symbol"]).upper(): float(r["pnl_net"]) for _, r in pnl_90d.iterrows()}

            print(f"\n  PnL {args.long_lookback_days}d por par:")
            for sym in current_symbols:
                pnl = pnl_map_90d.get(sym, 0.0)
                emoji = "🟢" if pnl > 0 else "🔴"
                print(f"    {emoji} {sym}: {pnl:+.2f} USD")

            # Encontrar el peor
            worst_candidates = [(sym, pnl_map_90d.get(sym, 0.0)) for sym in current_symbols]
            worst_candidates.sort(key=lambda x: x[1])
            if worst_candidates and worst_candidates[0][1] < 0:
                worst_symbol = worst_candidates[0][0]
                worst_pnl_90d = worst_candidates[0][1]
                print(f"\n  🗑 Peor performer: {worst_symbol} ({worst_pnl_90d:+.2f} USD)")
            else:
                print(f"\n  Todos los pares tienen PnL >= 0 en {args.long_lookback_days}d. No se elimina ninguno.")

        except Exception as exc:
            print(f"  ERROR evaluando 90d: {exc}")

    if worst_symbol:
        report["removed_pair"] = {"symbol": worst_symbol, "pnl_90d": worst_pnl_90d}
    else:
        report["removed_pair"] = None

    # ── Fase 5: Buscar reemplazo ────────────────────────────────────────
    print(f"\n═══ FASE 5: Buscar reemplazo por volumen ═══")
    replacement_entry: Optional[Dict[str, Any]] = None
    replacement_bt_pnl: float = 0.0

    if args.skip_rotation or worst_symbol is None:
        print("  (saltado: no hay par a reemplazar)")
    else:
        # Excluir pares actuales (menos el que se va a eliminar)
        remaining = {sym for sym in current_symbols if sym != worst_symbol}
        candidates = _fetch_top_volume_pairs(exclude=remaining, max_candidates=10)

        if not candidates:
            print("  No se encontraron candidatos en BingX API")
        else:
            print(f"  Top candidatos por volumen: {', '.join(candidates[:5])}")
            long_csv = REPO_ROOT / args.data_template

            for attempt, candidate_sym in enumerate(candidates[:args.max_replacement_attempts], 1):
                print(f"\n  --- Intento {attempt}/{args.max_replacement_attempts}: {candidate_sym} ---")

                # Descargar historial
                try:
                    ok = _download_full_history_for_symbol(
                        candidate_sym, long_csv, max_batches=args.max_history_batches,
                    )
                    if not ok:
                        print(f"  {candidate_sym}: sin historial disponible, saltando")
                        continue
                except Exception as exc:
                    print(f"  {candidate_sym}: error descargando historial: {exc}")
                    continue

                # Correr sweep
                candidate_best_path = out_dir / f"best_prod_replacement_{candidate_sym}.json"
                try:
                    _run_sweep_for_symbols(
                        [candidate_sym],
                        args.data_template,
                        args.sweep_cfg,
                        out_dir,
                        args.opt_lookback_days,
                        args.n_trials,
                        candidate_best_path,
                    )

                    if candidate_best_path.exists():
                        cand_entries = _load_best_entries(candidate_best_path)
                        if cand_entries:
                            replacement_entry = cand_entries[0]
                            # Evaluar para obtener PnL
                            try:
                                eval_res = _eval_symbol(
                                    candidate_best_path, candidate_sym,
                                    args.data_template, 1000.0, args.opt_lookback_days,
                                )
                                replacement_bt_pnl = eval_res.get("pnl_net", 0.0)
                            except Exception:
                                replacement_bt_pnl = 0.0
                            print(f"  ✅ {candidate_sym}: BT PnL={replacement_bt_pnl:+.2f} → SELECCIONADO")
                            break
                        else:
                            print(f"  ❌ {candidate_sym}: no paso filtros de backtesting")
                    else:
                        print(f"  ❌ {candidate_sym}: sweep no genero resultados")

                except Exception as exc:
                    print(f"  {candidate_sym}: error en sweep: {exc}")
                    continue

    if replacement_entry:
        report["replacement_pair"] = {
            "symbol": replacement_entry["symbol"],
            "bt_pnl": replacement_bt_pnl,
        }
    else:
        report["replacement_pair"] = None

    # ── Fase 6: Construir nuevo best_prod.json ──────────────────────────
    print(f"\n═══ FASE 6: Construir best_prod.json final ═══")
    final_entries: List[Dict[str, Any]] = []

    for sym in current_symbols:
        # Skip removed pair
        if worst_symbol and sym == worst_symbol and not args.skip_rotation:
            continue

        # Si fue re-optimizado, usar nuevos params
        if sym in new_params_map:
            final_entries.append(new_params_map[sym])
        # Si fue DISABLED (no paso backtesting), excluir
        elif any(e["symbol"] == sym and e.get("status") == "DISABLED" for e in evaluated_report):
            print(f"  Excluyendo {sym} (DISABLED)")
            continue
        else:
            # KEEP: mantener params originales
            final_entries.append(current_map[sym])

    # Agregar reemplazo
    if replacement_entry:
        final_entries.append(replacement_entry)

    report["final_pair_count"] = len(final_entries)
    report["final_symbols"] = [e["symbol"] for e in sorted(final_entries, key=lambda e: e["symbol"])]

    print(f"\n  Pares finales ({len(final_entries)}):")
    for e in sorted(final_entries, key=lambda e: e["symbol"]):
        tag = ""
        if e["symbol"] in new_params_map:
            tag = " (RE-OPTIMIZADO)"
        elif replacement_entry and e["symbol"] == replacement_entry["symbol"]:
            tag = " (NUEVO)"
        print(f"    • {e['symbol']}{tag}")

    # Escribir
    deployed = False
    if args.apply:
        print(f"\n  Aplicando cambios...")
        _save_best_entries(best_prod_path, final_entries, backup=True)
        report["applied"] = True

        # Tambien guardar copia en out_dir
        _save_best_entries(out_dir / "best_prod_final.json", final_entries, backup=False)
    else:
        print(f"\n  DRY RUN: No se modifica best_prod.json (usa --apply para aplicar)")
        _save_best_entries(out_dir / "best_prod_final.json", final_entries, backup=False)
        report["applied"] = False

    # ── Fase 7: Deploy ──────────────────────────────────────────────────
    if args.deploy:
        if not args.apply:
            print("\n  WARN: --deploy requiere --apply. Saltando deploy.")
            deployed = False
        else:
            print(f"\n═══ FASE 7: Desplegar a produccion ═══")
            # Detectar branch actual
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=str(REPO_ROOT), capture_output=True, text=True,
                )
                branch = result.stdout.strip() or "main"
            except Exception:
                branch = "main"
            print(f"  Branch: {branch}")
            deployed = _deploy_to_production(args.server, pem_key, branch)
    else:
        if args.apply:
            print("\n  INFO: Cambios aplicados localmente. Usa --deploy para desplegar a produccion.")

    report["deployed"] = deployed

    # ── Guardar reporte JSON ────────────────────────────────────────────
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"\n  Reporte guardado: {report_path}")

    # ── Telegram ────────────────────────────────────────────────────────
    if args.send_alert:
        print(f"\n═══ Enviando alerta Telegram ═══")
        msg = _format_summary(report)
        ok, detail = _send_telegram_if_possible(msg)
        if ok:
            print("  Telegram: enviado ✅")
        else:
            print(f"  Telegram: fallo ({detail})")

    # ── Resumen final ───────────────────────────────────────────────────
    print(f"\n{'═' * 50}")
    print(f"  EVALUACION COMPLETADA")
    print(f"  Pares: {len(current_symbols)} → {len(final_entries)}")
    print(f"  Aplicado: {'SI' if report.get('applied') else 'NO (dry run)'}")
    print(f"  Desplegado: {'SI' if deployed else 'NO'}")
    print(f"{'═' * 50}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
