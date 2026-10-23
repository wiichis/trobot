"""Bot BTC por rangos en modo PAPEL (22/10/2026) — no manda órdenes.

Combinación elegida por el usuario (v0.4 con reposición parcial, ver
`docs/btc_rangos_fase2_informe.md`): slots independientes (A), entrada E4 0,5 (caída de
0,5 × rango diario desde el máximo de 24 h), salida S4 0,75 (TP en +0,75 × rango, piso 0,6 %),
90 % de la ganancia queda en el bot y 10 % a holding, cierre mensual el día 3 con el DCA
reponiendo lo que alcance. Config en `pkg/btc_rangos.json`.

Paridad con la simulación por construcción: el job alimenta `pkg/btc_rangos_motor.Motor`, el
mismo motor del backtest, con las velas 1 h CERRADAS de BingX spot, una por una y en orden. Las
señales salen de `pkg/btc_rangos_reglas.py`. Los fills se simulan igual que en el backtest
(apertura de la vela siguiente, comisión 0,1 % + slippage 0,05 % por lado).

Cada hora (:01) procesa todas las velas cerradas posteriores a la última procesada, así que un
reinicio o una caída de pocas horas no pierden nada (ventana de 1000 velas). El estado
completo del motor vive en `archivos/btc_rangos/papel/estado.json` (escritura atómica). Si
existe pero no se puede leer, el job NO escribe nada y avisa (CRITICAL): perder el estado
significa perder los slots abiertos.

Salidas: `operaciones.csv`, `mensual.csv` (cierres mensuales), `equity.csv` (foto por corrida).
Avisos: `btc_rangos_inicio`, `btc_rangos_operaciones`, `btc_rangos_cierre_mensual`,
`btc_rangos_resumen` (diario), `btc_rangos_estado_ilegible`, `btc_rangos_modo_rechazado`.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from . import btc_rangos_motor as mm
from . import btc_rangos_reglas as rr

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "btc_rangos.json"
OUT_DIR = BASE_DIR / "archivos" / "btc_rangos" / "papel"
API = "https://open-api.bingx.com"
VENTANA_VELAS = 1000          # ~41 días: el rango de 14 días queda calculado en las últimas ~26
MIN_VELAS = 24 * 16
CAMPOS_COMBO = ("slots", "escalon_r", "entrada", "filtro", "salida", "ganancia", "reposicion_parcial",
                "dia_cierre", "capital")
_NO_SERIALIZAR = {"cfg", "dca", "slots", "operaciones", "meses"}


# --- config, fuentes ------------------------------------------------------------------
def _cfg() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def config_motor(cfg: Dict) -> mm.Config:
    return mm.Config(slots=cfg["slots"], escalon_r=float(cfg.get("escalon_r", 0.0)), salida=cfg["salida"],
                     ganancia=cfg["ganancia"], capital=float(cfg["capital"]),
                     reposicion_parcial=bool(cfg.get("reposicion_parcial", True)),
                     dia_cierre=int(cfg.get("dia_cierre", 3)), comision=float(cfg.get("comision", mm.COMISION)),
                     slippage=float(cfg.get("slippage", mm.SLIPPAGE)),
                     min_orden=float(cfg.get("min_orden_usdt", mm.MIN_ORDEN_USDT)))


def _velas_bingx(symbol: str, limit: int) -> pd.DataFrame:
    """Velas 1 h de BingX spot (la última puede estar en formación; se filtra después)."""
    r = requests.get(f"{API}/openApi/spot/v2/market/kline",
                     params={"symbol": symbol, "interval": "1h", "limit": limit}, timeout=15)
    d = r.json()
    if d.get("code") != 0:
        raise RuntimeError(f"klines spot code={d.get('code')} msg={d.get('msg')}")
    df = pd.DataFrame([k[:5] for k in d["data"]], columns=["ot", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ot"].astype("int64"), unit="ms")
    return df.drop_duplicates("date").set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)


def velas_cerradas(v: pd.DataFrame, ahora: pd.Timestamp) -> pd.DataFrame:
    """Sólo velas que ya cerraron (apertura + 1 h <= ahora). Índice naive en UTC."""
    ahora = ahora.tz_convert("UTC").tz_localize(None) if ahora.tzinfo is not None else ahora
    return v[v.index + pd.Timedelta(hours=1) <= ahora]


def diario_desde_1h(v: pd.DataFrame) -> pd.DataFrame:
    g = v.resample("1D")
    d = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return d[g["close"].count() == 24]


def preparar(v: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Señal, ATR previo y rango previo por vela, igual que el backtest (`mm.simular`)."""
    ind = rr.indicadores(v)
    ind["rango"] = rr.rango_diario(diario_desde_1h(v), v.index)
    out = v.copy()
    out["senal"] = rr.senal_entrada(v, ind, cfg["entrada"], bool(cfg.get("filtro", False)))
    out["atr_prev"] = ind["atr"].shift(1)
    out["rango_prev"] = ind["rango"].shift(1)
    out["rango"] = ind["rango"]
    return out


# --- estado ---------------------------------------------------------------------------
def _a_json(x):
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _a_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_a_json(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def motor_a_dict(m: mm.Motor) -> Dict:
    d = {f.name: _a_json(getattr(m, f.name)) for f in dataclasses.fields(m) if f.name not in _NO_SERIALIZAR}
    d["slots"] = [_a_json(dataclasses.asdict(s)) for s in m.slots]
    return d


def motor_desde_dict(d: Dict, cfg: mm.Config, dca: Callable[[int], float]) -> mm.Motor:
    slots = []
    for s in d["slots"]:
        s = dict(s)
        s["desde"] = pd.Timestamp(s["desde"]) if s.get("desde") else None
        slots.append(mm.Slot(**s))
    m = mm.Motor(cfg=cfg, dca=dca, slots=slots)
    for k, v in d.items():  # después de construir: __post_init__ reinicia el mes en curso
        if k != "slots":
            setattr(m, k, v)
    if m.mes.get("cierre"):
        m.mes["cierre"] = pd.Timestamp(m.mes["cierre"])
    return m


class EstadoIlegible(Exception):
    pass


def _rutas():
    return (OUT_DIR / "estado.json", OUT_DIR / "operaciones.csv", OUT_DIR / "mensual.csv", OUT_DIR / "equity.csv")


def leer_estado() -> Optional[Dict]:
    f = _rutas()[0]
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EstadoIlegible(str(exc)) from exc


def _guardar_estado(estado: Dict) -> None:
    f = _rutas()[0]
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(estado, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, f)


def _append_csv(f: Path, filas: List[Dict]) -> None:
    if filas:
        pd.DataFrame([_a_json(x) for x in filas]).to_csv(f, mode="a", header=not f.exists(), index=False)


# --- mensajes -------------------------------------------------------------------------
def _n(x: float, dec: int = 2) -> str:
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _signo(x: float, dec: int = 2) -> str:
    return ("+" if x >= 0 else "−") + _n(abs(x), dec)


def _emitir(categoria: str, severidad: str, **campos) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event(categoria, severidad, **campos)
    except Exception:
        pass


def objetivo_venta(cfg: Dict, s: mm.Slot, rango: float) -> float:
    p = mm.SALIDAS[cfg["salida"]]
    if p["tipo"] == "S1":
        return s.entrada * (1 + p["tp"])
    if p["tipo"] == "S4":
        r = mm.MIN_OBJETIVO if np.isnan(rango) else max(p["f"] * rango, mm.MIN_OBJETIVO)
        return s.entrada * (1 + r)
    return float("nan")


def mensaje_operaciones_md(eventos: List[Dict]) -> str:
    lineas = ["_Modo papel: sin dinero real._", ""]
    for e in eventos:
        if e["tipo"] == "compra":
            lineas.append(f"🟢 *Compra* slot {e['slot']} · {_n(e['qty'], 6)} BTC a {_n(e['precio'], 0)} "
                          f"({_n(e['usdt'])} USDT) · vende en {_n(e['objetivo'], 0)} o más")
        else:
            lineas.append(f"🔴 *Venta* slot {e['slot']} a {_n(e['precio'], 0)} · ganancia "
                          f"{_signo(e['ganancia'])} USDT ({_signo(e['pct'], 2)} %) en {e['horas']} h")
    return "\n".join(lineas)


def mensaje_cierre_md(mes: Dict) -> str:
    neto = mes["realizado"] - mes["atrapadas_costo"]
    return "\n".join([
        f"_Modo papel._ Ciclo {mes['ciclo']} cerrado a {_n(mes['precio_cierre'], 0)}.",
        f"Ganancia realizada {_signo(mes['realizado'])} USDT · {mes['ciclos']} ventas · {mes['entradas']} compras",
        f"Atrapados que pasan a holding: {mes['n_atrapadas']} (costo {_n(mes['atrapadas_costo'])} USDT)",
        f"*Neto del mes {_signo(neto)} USDT*",
        f"DCA {_n(mes['dca'], 0)} USDT: repone {_n(mes['repuesto'])} · compra holding {_n(mes['compra_holding'])}",
        f"Slots apagados: {mes['slots_apagados']}",
    ])


def mensaje_resumen_md(cfg: Dict, m: mm.Motor, precio: float, rango: float, ops: pd.DataFrame,
                       meses: pd.DataFrame, inicio: str) -> str:
    filas = ["```", "slot  estado     USDT   entrada  vende en"]
    for i, s in enumerate(m.slots, 1):
        if s.abierto:
            filas.append(f"{i:<5} comprado {s.costo:7.1f} {s.entrada:9.0f} {objetivo_venta(cfg, s, rango):9.0f}")
        elif s.activo:
            filas.append(f"{i:<5} libre    {s.nominal:7.1f}")
        else:
            filas.append(f"{i:<5} apagado  {max(s.nominal - s.costo, 0):7.1f}")
    filas.append("```")
    gan = float(ops["ganancia"].sum()) if len(ops) else 0.0
    atr = float(meses["atrapadas_costo"].sum()) if len(meses) else 0.0
    con_bot = m.patrimonio_btc(precio)
    sin_bot = m.bench_btc + m.cfg.capital / precio
    return "\n".join([
        f"_Modo papel desde {inicio[:10]}._ BTC {_n(precio, 0)} · rango diario {_n(rango * 100, 2)} %",
        *filas,
        f"Ventas: {len(ops)} · ganancia realizada {_signo(gan)} USDT · costo de atrapadas {_n(atr)} USDT",
        f"Capital del bot {_n(m.capital_bot_nominal())} USDT · holding {_n(m.holding_btc, 6)} BTC",
        f"BTC con bot {_n(con_bot, 6)} · sin bot {_n(sin_bot, 6)} ({_signo((con_bot / sin_bot - 1) * 100, 2)} %)",
    ])


# --- job ------------------------------------------------------------------------------
_AVISADO = set()


def run_btc_rangos(ahora: Optional[datetime] = None, fuentes: Optional[Dict[str, Callable]] = None) -> Optional[Dict]:
    """Job horario. Devuelve un resumen de la corrida, o None si no corrió. Nunca levanta."""
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return None
        modo = cfg.get("modo", "papel")
        if modo != "papel":
            if modo not in _AVISADO:
                _AVISADO.add(modo)
                log.warning("btc_rangos: modo %r no implementado; sólo 'papel'", modo)
                _emitir("btc_rangos_modo_rechazado", "WARN", modo=str(modo))
            return None
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        fuentes = fuentes or {}
        velas_fn = fuentes.get("velas") or (lambda limit: _velas_bingx(cfg.get("symbol", "BTC-USDT"), limit))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        f_estado, f_ops, f_mes, f_eq = _rutas()

        try:
            estado = leer_estado()
        except EstadoIlegible as exc:
            log.critical("btc_rangos: estado ilegible (%s); no se escribe nada", exc)
            _emitir("btc_rangos_estado_ilegible", "CRITICAL", detail=str(exc)[:300])
            return None

        v = velas_cerradas(velas_fn(VENTANA_VELAS), ahora)
        if len(v) < MIN_VELAS:
            log.warning("btc_rangos: %s velas cerradas (mínimo %s); se reintenta en 1 h", len(v), MIN_VELAS)
            return None
        datos = preparar(v, cfg)
        combo = {k: cfg.get(k) for k in CAMPOS_COMBO}
        dca_fn = lambda k: float(_cfg().get("dca_mensual_usdt", 0.0))  # se lee en el cierre: editable
        mcfg = config_motor(cfg)

        if estado is None:
            m = mm.Motor(cfg=mcfg, dca=dca_fn)
            estado = {"inicio": str(ahora), "modo": modo, "combo": combo, "ultima_barra": str(v.index[-1]),
                      "ultimo_resumen": None, "motor": motor_a_dict(m)}
            _guardar_estado(estado)
            _emitir("btc_rangos_inicio", "INFO", mensaje_md=(
                f"_Modo papel: sin dinero real._ Arranca con {_n(mcfg.capital, 0)} USDT en {mcfg.n_slots} slots. "
                f"Combinación {cfg['slots']} · {cfg['entrada']} · {cfg['salida']} · {cfg['ganancia']}. "
                f"Opera desde la vela que abre a las {(v.index[-1] + pd.Timedelta(hours=1)):%H:%M} UTC."))
            return {"nuevo": True, "procesadas": 0}
        if estado.get("combo") != combo:
            if "combo" not in _AVISADO:
                _AVISADO.add("combo")
                _emitir("btc_rangos_modo_rechazado", "WARN",
                        detail="la config cambió respecto del estado; no se mezclan combinaciones. "
                               "Para cambiar, archivar archivos/btc_rangos/papel/ y reiniciar.")
            return None

        m = motor_desde_dict(estado["motor"], mcfg, dca_fn)
        ultima = pd.Timestamp(estado["ultima_barra"])
        nuevas = datos[datos.index > ultima]
        if len(nuevas) and nuevas.index[0] > ultima + pd.Timedelta(hours=1):
            log.warning("btc_rangos: faltan velas entre %s y %s", ultima, nuevas.index[0])
        eventos, cierres = [], []
        for t, fila in nuevas.iterrows():
            a_entrar = list(m.pendientes)
            n_ops, n_meses = len(m.operaciones), len(m.meses)
            m.vela(t, fila["open"], fila["high"], fila["low"], fila["close"], bool(fila["senal"]),
                   float(fila["atr_prev"]), float(fila["rango_prev"]))
            for op in m.operaciones[n_ops:]:
                eventos.append({"tipo": "venta", "slot": op["slot"], "precio": op["salida"], "ganancia": op["ganancia"],
                                "pct": op["ganancia"] / op["costo"] * 100,
                                "horas": int((op["salida_t"] - op["entrada_t"]).total_seconds() // 3600)})
            for i in a_entrar:
                s = m.slots[i]
                if s.desde != t:
                    continue  # no entró (slot apagado o sin capital suficiente)
                op = next((o for o in m.operaciones[n_ops:] if o["slot"] == i + 1 and o["entrada_t"] == t), None)
                qty, entrada, usdt = (op["qty"], op["entrada"], op["costo"]) if op else (s.qty, s.entrada, s.costo)
                ref = mm.Slot(nominal=usdt, entrada=entrada)
                eventos.insert(len(eventos) - (1 if op else 0),
                               {"tipo": "compra", "slot": i + 1, "qty": qty, "precio": entrada, "usdt": usdt,
                                "objetivo": objetivo_venta(cfg, ref, float(fila["rango_prev"]))})
            cierres.extend(m.meses[n_meses:])
        _append_csv(f_ops, m.operaciones)
        _append_csv(f_mes, m.meses)
        m.operaciones, m.meses = [], []
        precio = float(v["close"].iloc[-1])
        rango_actual = float(datos["rango"].iloc[-1])
        _append_csv(f_eq, [{"ts": ahora, "precio": precio, "capital_bot": m.capital_bot_nominal(),
                            "usdt_bot": m.usdt_bot(), "btc_slots": m.btc_en_slots(), "holding_btc": m.holding_btc,
                            "bench_btc": m.bench_btc, "patrimonio_btc": m.patrimonio_btc(precio)}])
        if len(nuevas):
            estado["ultima_barra"] = str(nuevas.index[-1])
        estado["motor"] = motor_a_dict(m)
        estado["actualizado"] = str(ahora)

        if eventos:
            _emitir("btc_rangos_operaciones", "INFO", mensaje_md=mensaje_operaciones_md(eventos))
        for mes in cierres:
            _emitir("btc_rangos_cierre_mensual", "INFO", mensaje_md=mensaje_cierre_md(mes))
        hoy = str(ahora.date())
        if estado.get("ultimo_resumen") != hoy and ahora.hour >= int(cfg.get("hora_resumen_utc", 12)):
            ops = pd.read_csv(f_ops) if f_ops.exists() else pd.DataFrame()
            meses = pd.read_csv(f_mes) if f_mes.exists() else pd.DataFrame()
            _emitir("btc_rangos_resumen", "INFO",
                    mensaje_md=mensaje_resumen_md(cfg, m, precio, rango_actual, ops, meses, estado["inicio"]))
            estado["ultimo_resumen"] = hoy
        _guardar_estado(estado)
        return {"nuevo": False, "procesadas": len(nuevas), "eventos": len(eventos), "cierres": len(cierres)}
    except Exception as exc:  # el scheduler es de un solo hilo: nunca propagar
        log.exception("btc_rangos: error en la corrida: %s", exc)
        return None
