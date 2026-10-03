"""Cuatro bots sobre BTC con libros virtuales y UNA posición neta (02/10/2026).

Configuración en `pkg/btc_bots.json`; reglas en `pkg/btc_reglas.py`, las mismas que usa la
simulación histórica (`scripts/btc_cuatro_bots.py`). Resultado en `archivos/btc_bots/`:
- `estado.json`      libro de cada bot (efectivo, posición, funding, comisiones).
- `operaciones.csv`  cada cambio de posición de cada bot.
- `equity.csv`       foto horaria del capital de cada bot y de la posición neta.

Diseño:
- Una sola cuenta de BingX: cada bot lleva su libro virtual y al exchange iría la SUMA de
  sus posiciones en el perpetuo (la orden neta). En modo `papel` (SIN DINERO) no se manda
  ninguna orden: se registra la posición neta que habría que tener. El modo `real` todavía
  no está implementado y se rechaza.
- Job HORARIO (:02). Cada bot decide una vez por vela cerrada de su temporalidad (A y D
  diaria, B de 4 h) con la posición que da su regla en la última vela cerrada. El estado
  en disco evita decidir dos veces la misma vela y sobrevive a reinicios. Con 500 velas
  la regla decide igual que con toda la historia (verificado sobre 2017-2026).
- Ejecución al precio vivo del perpetuo, con `costo_lado` por lado (taker + slippage,
  igual que la simulación). C paga además `costo_spot_lado` en el contado.
- Funding: cada liquidación (cada 8 h) se paga/cobra sobre la posición de cada bot con
  la tasa y el precio de referencia que publica BingX. Se procesa ANTES de decidir, así
  que se aplica a la posición que estaba abierta en ese momento.
- C (cobro de funding): compra BTC al contado por `exposicion` × capital y abre un short
  del perpetuo por la misma cantidad; el precio se compensa y queda el funding.
- Corte: si el capital de un bot cae `corte_perdida` bajo el inicial, cierra y se apaga.
- NUNCA levanta excepción: en `main.py` una excepción en un job corta el loop entero.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests

from .btc_reglas import posiciones

log = logging.getLogger("trobot")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "btc_bots.json"
OUT_DIR = BASE_DIR / "archivos" / "btc_bots"
API = "https://open-api.bingx.com"
QTY_STEP = 0.0001          # BTC-USDT perpetuo (endpoint de contratos, 02/10/2026)
VENTANA_VELAS = 500
MIN_VELAS = 120
HORAS = {"1d": 24, "4h": 4}
LADO = {1: "LONG", -1: "SHORT", 0: "fuera"}


def _cfg() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rutas():
    return OUT_DIR / "estado.json", OUT_DIR / "operaciones.csv", OUT_DIR / "equity.csv"


def _n(x: float, dec: int = 2) -> str:
    """Número con formato español: 84.464,40."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------- datos públicos de BingX

def _velas_bingx(symbol: str, interval: str, limit: int) -> List[Dict]:
    from .price_bingx_5m import _fetch_bingx_candles
    return _fetch_bingx_candles(symbol, limit, interval=interval)


def _precio_bingx(symbol: str) -> Optional[float]:
    r = requests.get(f"{API}/openApi/swap/v2/quote/price", params={"symbol": symbol}, timeout=8)
    data = r.json().get("data", {})
    if isinstance(data, list):
        data = data[0] if data else {}
    return float(data["price"])


def _funding_bingx(symbol: str) -> List[Dict]:
    r = requests.get(f"{API}/openApi/swap/v2/quote/fundingRate", params={"symbol": symbol, "limit": 100}, timeout=8)
    eventos = [{"ms": int(x["fundingTime"]), "tasa": float(x["fundingRate"]), "precio": float(x["markPrice"])}
               for x in (r.json().get("data") or [])]
    return sorted(eventos, key=lambda e: e["ms"])


def velas_cerradas(velas: List[Dict], horas: int, ahora: pd.Timestamp) -> pd.DataFrame:
    """Sólo velas CERRADAS, indexadas por su apertura (UTC)."""
    if not velas:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    d = pd.DataFrame(velas)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    d = d[d["date"] + pd.Timedelta(hours=horas) <= ahora]
    return d.drop_duplicates("date").set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)


# ---------------------------------------------------------------- libros virtuales

def libro_nuevo(capital: float) -> Dict:
    return {"capital_inicial": capital, "efectivo": capital, "lado": 0, "qty": 0.0,
            "precio_entrada": None, "desde_ms": None, "spot_qty": 0.0, "ultima_barra": None,
            "acum_posicion": 0.0, "funding_neto": 0.0, "comisiones": 0.0, "operaciones": 0,
            "apagado": False}


def capital(libro: Dict, precio: float) -> float:
    abierta = libro["lado"] * libro["qty"] * (precio - libro["precio_entrada"]) if libro["lado"] else 0.0
    return libro["efectivo"] + libro["spot_qty"] * precio + abierta


def redondear_qty(q: float) -> float:
    return round(round(q / QTY_STEP) * QTY_STEP, 4)


def _cerrar_perpetuo(libro: Dict, precio: float, costo_lado: float) -> Optional[float]:
    """Cierra la posición del perpetuo. Devuelve el resultado neto de la posición
    (precio, comisiones de entrada y salida y funding cobrado/pagado mientras estuvo)."""
    if not libro["lado"]:
        return None
    pnl = libro["lado"] * libro["qty"] * (precio - libro["precio_entrada"])
    com = libro["qty"] * precio * costo_lado
    libro["efectivo"] += pnl - com
    libro["comisiones"] += com
    resultado = pnl - com + libro["acum_posicion"]
    libro.update(lado=0, qty=0.0, precio_entrada=None, desde_ms=None, acum_posicion=0.0)
    return resultado


def _abrir_perpetuo(libro: Dict, lado: int, qty: float, precio: float, costo_lado: float, ms: int) -> None:
    com = qty * precio * costo_lado
    libro["efectivo"] -= com
    libro["comisiones"] += com
    libro.update(lado=lado, qty=qty, precio_entrada=precio, desde_ms=ms, acum_posicion=-com)


def cambiar_posicion(libro: Dict, nuevo_lado: int, precio: float, costo_lado: float,
                     exposicion: float, ms: int) -> Dict:
    antes = libro["lado"]
    resultado = _cerrar_perpetuo(libro, precio, costo_lado)
    qty = 0.0
    if nuevo_lado:
        qty = redondear_qty(exposicion * capital(libro, precio) / precio)
        if qty > 0:
            _abrir_perpetuo(libro, nuevo_lado, qty, precio, costo_lado, ms)
    libro["operaciones"] += 1
    if antes and nuevo_lado:
        accion = f"da vuelta: cierra {LADO[antes]}, abre {LADO[nuevo_lado]}"
    elif nuevo_lado:
        accion = f"abre {LADO[nuevo_lado]}"
    else:
        accion = f"cierra {LADO[antes]}"
    if nuevo_lado and qty == 0:
        accion += " (capital insuficiente para el mínimo de 0,0001 BTC)"
    return {"accion": accion, "lado_antes": antes, "lado_despues": libro["lado"], "qty": qty,
            "resultado_cerrada": resultado}


def abrir_carry(libro: Dict, precio: float, costo_lado: float, costo_spot: float,
                exposicion: float, ms: int) -> Dict:
    qty = redondear_qty(exposicion * libro["efectivo"] / precio)
    com_spot = qty * precio * costo_spot
    libro["efectivo"] -= qty * precio + com_spot
    libro["comisiones"] += com_spot
    libro["spot_qty"] = qty
    _abrir_perpetuo(libro, -1, qty, precio, costo_lado, ms)
    libro["acum_posicion"] -= com_spot
    libro["operaciones"] += 1
    return {"accion": f"compra {_n(qty, 4)} BTC al contado + short del mismo tamaño",
            "lado_antes": 0, "lado_despues": -1, "qty": qty, "resultado_cerrada": None}


def cerrar_todo(libro: Dict, precio: float, costo_lado: float, costo_spot: float) -> Optional[float]:
    resultado = _cerrar_perpetuo(libro, precio, costo_lado)
    if libro["spot_qty"]:
        com = libro["spot_qty"] * precio * costo_spot
        libro["efectivo"] += libro["spot_qty"] * precio - com
        libro["comisiones"] += com
        libro["spot_qty"] = 0.0
        resultado = (resultado or 0.0) - com
    return resultado


def aplicar_funding(libros: Dict[str, Dict], eventos: List[Dict], desde_ms: int) -> int:
    """Paga/cobra cada liquidación posterior a `desde_ms` sobre las posiciones abiertas
    ANTES de esa liquidación. El long paga si la tasa es positiva; el short cobra."""
    ultimo = desde_ms
    for ev in eventos:
        if ev["ms"] <= desde_ms:
            continue
        for libro in libros.values():
            if libro["lado"] and libro["qty"] > 0 and (libro["desde_ms"] or 0) <= ev["ms"]:
                pago = libro["lado"] * libro["qty"] * ev["precio"] * ev["tasa"]
                libro["efectivo"] -= pago
                libro["funding_neto"] -= pago
                libro["acum_posicion"] -= pago
        ultimo = max(ultimo, ev["ms"])
    return ultimo


# ---------------------------------------------------------------- persistencia y avisos

def _guardar_estado(f: Path, estado: Dict) -> None:
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)


def _append_csv(f: Path, filas: List[Dict]) -> None:
    if filas:
        pd.DataFrame(filas).to_csv(f, mode="a", header=not f.exists(), index=False)


def _emitir(categoria: str, severidad: str, **campos) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event(categoria, severidad, **campos)
    except Exception:
        pass


_MODO_AVISADO = set()


def _rechazar_modo(modo) -> None:
    if modo not in _MODO_AVISADO:
        _MODO_AVISADO.add(modo)
        log.warning("btc_bots: modo %r no soportado; sólo 'papel' está implementado", modo)
        _emitir("btc_bots_modo_rechazado", "WARN", modo=str(modo),
                detalle="sólo el modo papel (sin dinero) está implementado; los bots no corren")


def _texto_capitales(cfg: Dict, estado: Dict, precio: float) -> str:
    partes, total, inicial = [], 0.0, 0.0
    for k, b in cfg["bots"].items():
        libro = estado["bots"].get(k)
        if not libro:
            continue
        c = capital(libro, precio)
        total += c
        inicial += libro["capital_inicial"]
        estado_txt = "apagado" if libro["apagado"] else LADO[libro["lado"]]
        partes.append(f"{k} {b['nombre']}: {_n(c)} ({_n((c / libro['capital_inicial'] - 1) * 100, 1)}%, {estado_txt})")
    partes.append(f"Total: {_n(total)} de {_n(inicial)} ({_n((total / inicial - 1) * 100, 1)}%)")
    return "\n".join(partes)


def _signo(x: float, dec: int = 1) -> str:
    if round(abs(x), dec) == 0:
        return _n(0.0, dec)
    return ("+" if x > 0 else "-") + _n(abs(x), dec)


def tabla_md(cfg: Dict, estado: Dict, precio: float) -> str:
    """Capital de cada bot en un bloque de ancho fijo (se alinea en Telegram)."""
    filas = [f"{'Bot':<10}{'Capital':>8}{'Var.':>8}  Posición"]
    total = inicial = 0.0
    for k, b in cfg["bots"].items():
        libro = estado["bots"].get(k)
        if not libro:
            continue
        c = capital(libro, precio)
        total += c
        inicial += libro["capital_inicial"]
        pos = "apagado" if libro["apagado"] else ("cobro" if libro["spot_qty"] else LADO[libro["lado"]])
        var = _signo((c / libro["capital_inicial"] - 1) * 100) + "%"
        filas.append(f"{k + ' ' + b.get('corto', b['nombre']):<10}{_n(c):>8}{var:>8}  {pos}")
    if inicial:
        filas.append(f"{'Total':<10}{_n(total):>8}{_signo((total / inicial - 1) * 100) + '%':>8}")
    return "```\n" + "\n".join(filas) + "\n```"


def _emoji_operacion(accion: str) -> str:
    if accion.startswith("CORTE"):
        return "🛑"
    if accion.startswith("da vuelta"):
        return "🔄"
    if "contado" in accion:
        return "⚖️"
    if accion.startswith("abre LONG"):
        return "🟢"
    if accion.startswith("abre SHORT"):
        return "🔴"
    return "⚪"


def _modo_md(modo: str) -> str:
    return "_Simulación SIN DINERO_" if modo == "papel" else f"_Modo {modo}_"


def mensaje_operaciones_md(cfg: Dict, estado: Dict, filas_ops: List[Dict], precio: float, neta: float, modo: str) -> str:
    from .lifecycle_events import escapar_md
    lineas = [_modo_md(modo)]
    for f in filas_ops:
        linea = f"{_emoji_operacion(f['accion'])} *{f['bot']} · {f['nombre']}* — {escapar_md(f['accion'])}"
        if f["resultado_cerrada"] is not None:
            linea += f"\n      resultado `{_signo(f['resultado_cerrada'], 2)} USDT`"
        lineas.append(linea)
    lineas += [f"Precio BTC `{_n(precio, 1)}`", "", tabla_md(cfg, estado, precio),
               f"Posición neta en el perpetuo: `{_signo(neta, 4)} BTC`"]
    return "\n".join(lineas)


def mensaje_resumen_md(cfg: Dict, estado: Dict, precio: float, neta: float, modo: str,
                       ventana: Optional[str], inicio: bool = False) -> str:
    if inicio:
        cap = _n(next(iter(estado["bots"].values()))["capital_inicial"]) if estado["bots"] else "0"
        cabecera = [f"Arrancan los 4 bots · {_modo_md(modo)}",
                    f"Cada uno con `{cap} USDT`. Avisan cuando operan y mandan un resumen diario."]
    else:
        cabecera = [f"{_modo_md(modo)} · BTC `{_n(precio, 1)}`"]
    lineas = cabecera + ["", tabla_md(cfg, estado, precio), f"Posición neta en el perpetuo: `{_signo(neta, 4)} BTC`"]
    if ventana:
        lineas += ["", f"📈 Ventana BTC: {ventana}"]
    return "\n".join(lineas)


# ---------------------------------------------------------------- job

def run_btc_bots(ahora: Optional[datetime] = None, fuentes: Optional[Dict[str, Callable]] = None) -> Optional[Dict]:
    """Job horario. Devuelve un resumen del ciclo, o None si no corrió o falló."""
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return None
        modo = cfg.get("modo", "papel")
        if modo != "papel":
            _rechazar_modo(modo)
            return None
        symbol = cfg.get("symbol", "BTC-USDT")
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        ms = int(ahora.timestamp() * 1000)
        fuentes = fuentes or {}
        velas_fn = fuentes.get("velas") or (lambda interval, limit: _velas_bingx(symbol, interval, limit))
        precio_fn = fuentes.get("precio") or (lambda: _precio_bingx(symbol))
        funding_fn = fuentes.get("funding") or (lambda: _funding_bingx(symbol))
        costo, costo_spot = float(cfg["costo_lado"]), float(cfg["costo_spot_lado"])
        corte = float(cfg.get("corte_perdida", 0.30))

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f_estado, f_ops, f_eq = _rutas()
        estado = json.loads(f_estado.read_text(encoding="utf-8")) if f_estado.exists() else None

        precio = precio_fn()
        if not precio or precio <= 0:
            log.warning("btc_bots: sin precio vivo; se reintenta en 1 h")
            return None
        try:
            eventos = funding_fn() or []
        except Exception as exc:
            log.warning("btc_bots: sin historial de funding (%s); se aplica en el próximo ciclo", exc)
            eventos = []

        nuevo = estado is None
        if nuevo:
            estado = {"inicio": str(ahora), "modo": modo,
                      "ultimo_funding_ms": max([e["ms"] for e in eventos], default=ms),
                      "bots": {}, "ultimo_resumen": str(ahora.date())}
        else:
            estado["ultimo_funding_ms"] = aplicar_funding(estado["bots"], eventos, int(estado["ultimo_funding_ms"]))

        velas, ops = {}, []
        for k, b in cfg["bots"].items():
            libro = estado["bots"].setdefault(k, libro_nuevo(float(cfg["capital_total"]) * float(b["peso"])))
            if libro["apagado"]:
                continue
            exp = float(b["exposicion"])
            if b["regla"] == "carry":
                if not libro["lado"] and not libro["spot_qty"]:
                    ops.append({"bot": k, "barra": None, **abrir_carry(libro, precio, costo, costo_spot, exp, ms)})
                continue
            tf = b["temporalidad"]
            if tf not in velas:
                velas[tf] = velas_cerradas(velas_fn(tf, VENTANA_VELAS), HORAS[tf], ahora)
            v = velas[tf]
            if len(v) < MIN_VELAS:
                log.warning("btc_bots: %s velas %s cerradas para el bot %s; se reintenta", len(v), tf, k)
                continue
            barra = str(v.index[-1])
            if barra == libro["ultima_barra"]:
                continue
            objetivo = int(posiciones(v, b["regla"], bool(b.get("solo_long", False))).iat[-1])
            libro["ultima_barra"] = barra
            if objetivo != libro["lado"]:
                ops.append({"bot": k, "barra": barra, **cambiar_posicion(libro, objetivo, precio, costo, exp, ms)})

        for k, libro in estado["bots"].items():
            if not libro["apagado"] and capital(libro, precio) <= (1 - corte) * libro["capital_inicial"]:
                antes = libro["lado"]
                resultado = cerrar_todo(libro, precio, costo, costo_spot)
                libro["apagado"] = True
                ops.append({"bot": k, "barra": None, "accion": f"CORTE: perdió {int(corte * 100)}% o más; cierra y se apaga",
                            "lado_antes": antes, "lado_despues": 0, "qty": 0.0, "resultado_cerrada": resultado})

        neta = round(sum(l["lado"] * l["qty"] for l in estado["bots"].values()), 4)
        estado["posicion_neta_btc"] = neta
        estado["actualizado"] = str(ahora)
        _guardar_estado(f_estado, estado)

        filas_ops = [{"ts": str(ahora), "bot": o["bot"], "nombre": cfg["bots"].get(o["bot"], {}).get("nombre", ""),
                      "accion": o["accion"], "lado_antes": o["lado_antes"], "lado_despues": o["lado_despues"],
                      "qty": o["qty"], "precio": precio, "resultado_cerrada": o["resultado_cerrada"],
                      "capital_bot": capital(estado["bots"][o["bot"]], precio), "barra": o["barra"], "modo": modo}
                     for o in ops]
        _append_csv(f_ops, filas_ops)
        foto = {"ts": str(ahora), "precio": precio, "modo": modo, "posicion_neta_btc": neta}
        foto.update({k: round(capital(l, precio), 4) for k, l in estado["bots"].items()})
        foto["total"] = round(sum(capital(l, precio) for l in estado["bots"].values()), 4)
        _append_csv(f_eq, [foto])

        modo_txt = "SIN DINERO (papel)" if modo == "papel" else modo
        if nuevo:
            _emitir("btc_bots_inicio", "INFO", modo=modo_txt, precio_btc=_n(precio, 1),
                    capital=_texto_capitales(cfg, estado, precio),
                    mensaje_md=mensaje_resumen_md(cfg, estado, precio, neta, modo, None, inicio=True))
        if ops:
            lineas = []
            for f in filas_ops:
                linea = f"{f['bot']} {f['nombre']}: {f['accion']} @ {_n(precio, 1)}"
                if f["resultado_cerrada"] is not None:
                    linea += f" | resultado {_n(f['resultado_cerrada'], 2)} USDT"
                lineas.append(linea)
            _emitir("btc_bots_operaciones", "INFO", modo=modo_txt, operaciones="\n".join(lineas),
                    posicion_neta=f"{_n(neta, 4)} BTC", capital=_texto_capitales(cfg, estado, precio),
                    mensaje_md=mensaje_operaciones_md(cfg, estado, filas_ops, precio, neta, modo))
        if not nuevo and estado.get("ultimo_resumen") != str(ahora.date()):
            estado["ultimo_resumen"] = str(ahora.date())
            _guardar_estado(f_estado, estado)
            ventana = ventana_md = None
            try:
                from .btc_alertas import leer_estado, linea_md, texto_estado
                ev = leer_estado()
                if ev:
                    ventana, ventana_md = texto_estado(ev), linea_md(ev)
            except Exception:
                pass
            _emitir("btc_bots_resumen", "INFO", modo=modo_txt, precio_btc=_n(precio, 1),
                    capital=_texto_capitales(cfg, estado, precio), posicion_neta=f"{_n(neta, 4)} BTC",
                    ventana_btc=ventana,
                    mensaje_md=mensaje_resumen_md(cfg, estado, precio, neta, modo, ventana_md))
        return {"nuevo": nuevo, "operaciones": filas_ops, "posicion_neta_btc": neta, "total": foto["total"]}
    except Exception as exc:   # nunca tirar el bot por los bots de BTC
        log.warning("btc_bots falló: %s", exc)
        return None
