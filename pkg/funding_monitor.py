"""Monitor del carry de funding (09/10/2026). NO manda órdenes.

Mide a diario el funding anualizado medio (trailing 30 días) de los N pares de BingX con
mayor funding positivo, entre los líquidos que además tienen contado en BingX con volumen
suficiente para ejecutar la pata spot (el contado de BingX es fino: los micro-caps de funding
30-80% mueven ~50 mil USDT/día y no se pueden operar; el backtest los excluía por construcción). Avisa por
Telegram cuando esa métrica entra en el régimen ALTO (>= 20%) o sale de él (< 15%), y una
vez al arrancar con el estado actual.

Por qué esos umbrales (estudio 09/10, `scripts/funding_carry_regimen.py`, semanal, 10 pares,
costos 0,17% por entrada o salida): con la métrica >= 20% el carry rindió ~17-20%/año sobre el
capital total; entre 10% y 20%, ~3%; bajo 10%, ~0%. El carry es un efecto del apetito de
riesgo del mercado: paga cuando los longs apalancados dominan. En 2025-26 casi no se ha dado.

Estado en `archivos/funding/monitor_estado.json`, historial en `monitor_log.csv`. NUNCA levanta
excepción. Se evalúa una vez por día UTC (después de la 01:00, con el funding de las 00:00).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

log = logging.getLogger("trobot")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "funding_monitor.json"
OUT_DIR = BASE_DIR / "archivos" / "funding"
API = "https://open-api.bingx.com"
DIA_MS = 86_400_000
NOTA = "Aviso de un régimen medido (2019-2026); no es una recomendación."


def _n(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _get(path: str, **params):
    r = requests.get(f"{API}/openApi/{path}", params=params, timeout=10)
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"{path}: {j.get('msg')}")
    return j.get("data")


def _tickers() -> List[Dict]:
    return _get("swap/v2/quote/ticker") or []


def _volumen_spot() -> Dict[str, float]:
    """Volumen 24h en USDT del contado de BingX, por símbolo (la pata spot debe poder ejecutarse)."""
    return {t["symbol"]: float(t.get("quoteVolume", 0) or 0) for t in (_get("spot/v1/ticker/24hr") or [])}


def _funding(symbol: str) -> List[Dict]:
    d = _get("swap/v2/quote/fundingRate", symbol=symbol, limit=250) or []
    return sorted(({"ms": int(x["fundingTime"]), "tasa": float(x["fundingRate"])} for x in d), key=lambda e: e["ms"])


def funding_anualizado(eventos: List[Dict], ahora_ms: int, dias: int) -> Optional[float]:
    """Suma de las tasas liquidadas en la ventana, anualizada. None si la historia no cubre la ventana."""
    ini = ahora_ms - dias * DIA_MS
    if not eventos or eventos[0]["ms"] > ini + 2 * DIA_MS:
        return None
    suma = sum(e["tasa"] for e in eventos if e["ms"] >= ini)
    return suma / dias * 365


def nivel(metrica: float, previo: Optional[str], cfg: Dict) -> str:
    """alto / medio / bajo con histéresis: se entra a alto en `umbral_alto` y se sale bajo `umbral_salida`."""
    if metrica >= cfg["umbral_alto"] or (previo == "alto" and metrica >= cfg["umbral_salida"]):
        return "alto"
    return "medio" if metrica >= cfg["umbral_medio"] else "bajo"


def medir(cfg: Dict, ahora: pd.Timestamp, fuentes: Dict[str, Callable]) -> Dict:
    tickers = sorted(fuentes["tickers"](), key=lambda t: -float(t.get("quoteVolume", 0) or 0))
    spot = fuentes["spot"]()
    elegibles = [t["symbol"] for t in tickers
                 if spot.get(t["symbol"], 0) >= cfg["min_volumen_spot_usdt_24h"]
                 and float(t.get("quoteVolume", 0) or 0) >= cfg["min_volumen_usdt_24h"]][:cfg["universo"]]
    ahora_ms = int(ahora.timestamp() * 1000)
    medidos = {}
    for sym in elegibles:
        try:
            m = funding_anualizado(fuentes["funding"](sym), ahora_ms, cfg["lookback_dias"])
        except Exception as exc:
            log.warning("funding_monitor: %s falló (%s)", sym, exc)
            continue
        if m is not None:
            medidos[sym] = m
        if fuentes.get("pausa"):
            fuentes["pausa"]()
    positivos = sorted(((s, m) for s, m in medidos.items() if m > 0), key=lambda x: -x[1])[:cfg["n_pares"]]
    metrica = sum(m for _, m in positivos) / len(positivos) if positivos else 0.0
    mediana = float(pd.Series(list(medidos.values())).median()) if medidos else 0.0
    return {"fecha": str(ahora.date()), "metrica": metrica, "mediana_mercado": mediana,
            "n_medidos": len(medidos), "top": [{"par": s, "anual": m} for s, m in positivos]}


def mensaje_md(prev: Optional[Dict], e: Dict, cfg: Dict) -> str:
    est = e["nivel"]
    cabecera = {"alto": "🟢 *Carry de funding en régimen ALTO*",
                "medio": "🟡 *Carry de funding en régimen MEDIO*",
                "bajo": "⚪ *Carry de funding en régimen BAJO*"}[est]
    if prev is None:
        cabecera = "*Estado actual* · " + cabecera
    elif prev["nivel"] == "alto" and est != "alto":
        cabecera = "🔴 *El carry de funding salió del régimen alto*"
    lineas = [cabecera, "",
              f"Funding medio top {len(e['top'])}   `{_n(e['metrica'] * 100, 1)}%` anual (30 días)",
              f"Mediana del mercado        `{_n(e['mediana_mercado'] * 100, 1)}%` anual",
              f"Umbral alto / salida       `{_n(cfg['umbral_alto'] * 100)}%` / `{_n(cfg['umbral_salida'] * 100)}%`",
              "", "```"]
    for x in e["top"]:
        lineas.append(f"{x['par'].replace('-USDT', ''):<10}{_n(x['anual'] * 100, 1):>7}%")
    lineas += ["```", "",
               "Histórico (semanal, 10 pares, con costos): métrica ≥20% → ~17-20%/año sobre el capital; "
               "10-20% → ~3%; bajo 10% → ~0%. Hace falta contado + short del perpetuo.",
               "", f"_{NOTA}_"]
    return "\n".join(lineas)


def _rutas():
    return OUT_DIR / "monitor_estado.json", OUT_DIR / "monitor_log.csv"


def leer_estado() -> Optional[Dict]:
    try:
        f, _ = _rutas()
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    except Exception:
        return None


def _emitir(**campos) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event("funding_monitor", "INFO", **campos)
    except Exception:
        pass


def run_funding_monitor(ahora: Optional[datetime] = None, fuentes: Optional[Dict[str, Callable]] = None) -> Optional[Dict]:
    """Job horario. Devuelve {} si no hubo aviso, los campos del aviso si avisó, None si no corrió."""
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not cfg.get("enabled", True):
            return None
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        if ahora.hour < 1:
            return None
        prev = leer_estado()
        if prev is not None and prev.get("fecha") == str(ahora.date()):
            return {}
        src = {"tickers": _tickers, "spot": _volumen_spot, "funding": _funding, "pausa": lambda: time.sleep(0.05)}
        src.update(fuentes or {})
        e = medir(cfg, ahora, src)
        if e["n_medidos"] < 20 or not e["top"]:
            log.warning("funding_monitor: sólo %s pares medidos; se reintenta", e["n_medidos"])
            return None
        e["nivel"] = nivel(e["metrica"], prev["nivel"] if prev else None, cfg)
        avisa = prev is None or (e["nivel"] == "alto") != (prev["nivel"] == "alto")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f_estado, f_log = _rutas()
        tmp = f_estado.with_suffix(".tmp")
        tmp.write_text(json.dumps(e, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, f_estado)
        pd.DataFrame([{"fecha": e["fecha"], "nivel": e["nivel"], "metrica": e["metrica"], "mediana": e["mediana_mercado"],
                       "n_medidos": e["n_medidos"], "top": ";".join(f"{x['par']}:{x['anual']:.3f}" for x in e["top"])}]) \
            .to_csv(f_log, mode="a", header=not f_log.exists(), index=False)
        if not avisa:
            return {}
        campos = {"nivel": e["nivel"], "metrica_pct": _n(e["metrica"] * 100, 1), "nota": NOTA}
        _emitir(**campos, mensaje_md=mensaje_md(prev, e, cfg))
        return campos
    except Exception as exc:   # nunca tirar el bot por el monitor
        log.warning("funding_monitor falló: %s", exc)
        return None


_HILO: Optional[threading.Thread] = None


def run_funding_monitor_async() -> None:
    """Para el scheduler: la medición hace ~90 pedidos HTTP (decenas de segundos) y `schedule`
    corre todo en un solo hilo, así que correrla en línea retrasaría el job que mueve los
    stops. Va en un hilo aparte, de a uno."""
    global _HILO
    if _HILO is not None and _HILO.is_alive():
        return
    _HILO = threading.Thread(target=run_funding_monitor, name="funding_monitor", daemon=True)
    _HILO.start()
