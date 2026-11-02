"""Bot BTC por rangos en modo PAPEL (22/10/2026) — no manda órdenes.

Desde el 02/11/2026 corre con las reglas de lotes, "reponer sólo cuando hay dinero"
(`pkg/btc_rangos_lotes.py`, 3 ciclos por lote; informe en `docs/btc_rangos_lotes_informe.md`),
como sombra del bot real (`pkg/btc_rangos_real.py`): mismo capital y los mismos depósitos, que
lee de `archivos/btc_rangos/real/depositos.csv` y aplica a la apertura de la vela en que el real
los detectó. La diferencia entre los dos es el costo de la ejecución real. Entrada E4 0,5, salida
S4 0,75 con piso de costo, 90 % de la ganancia queda en el bot y 10 % a holding. Config en
`pkg/btc_rangos.json`. El estado v0.4 se migra solo en la primera corrida, con respaldo en
`archivos/btc_rangos/papel_v04_<fecha>/`.

Paridad con la simulación por construcción: el job alimenta `MotorLotes.vela`, el mismo paso de
`scripts/btc_rangos_lotes_sim.py`, con las velas 1 h CERRADAS de BingX spot, una por una y en
orden. Las señales salen de `pkg/btc_rangos_reglas.py`. Los fills se simulan igual que en el
backtest (apertura de la vela siguiente, comisión 0,1 % + slippage 0,05 % por lado). El día 3
es sólo un corte contable.

Cada hora (:01) procesa todas las velas cerradas posteriores a la última procesada, así que un
reinicio o una caída de pocas horas no pierden nada (ventana de 1000 velas). El estado
completo del motor vive en `archivos/btc_rangos/papel/estado.json` (escritura atómica). Si
existe pero no se puede leer, el job NO escribe nada y avisa (CRITICAL): perder el estado
significa perder los slots abiertos.

Salidas: `operaciones.csv`, `mensual.csv` (cortes del día 3), `equity.csv` (foto por corrida).
Avisos: `btc_rangos_inicio`, `btc_rangos_operaciones`, `btc_rangos_cierre_mensual`,
`btc_rangos_resumen` (diario), `btc_rangos_estado_ilegible`, `btc_rangos_modo_rechazado`.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from . import btc_rangos_lotes as ml
from . import btc_rangos_motor as mm
from . import btc_rangos_reglas as rr

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "btc_rangos.json"
OUT_DIR = BASE_DIR / "archivos" / "btc_rangos" / "papel"
DEPOSITOS_REAL = BASE_DIR / "archivos" / "btc_rangos" / "real" / "depositos.csv"
API = "https://open-api.bingx.com"
VENTANA_VELAS = 1000          # ~41 días: el rango de 14 días queda calculado en las últimas ~26
MIN_VELAS = 24 * 16
CAMPOS_COMBO = ("reglas", "ciclos_por_lote", "caida_atrapado", "dias_atrapado", "entrada", "filtro", "salida",
                "ganancia", "dia_cierre", "capital")


# --- config, fuentes ------------------------------------------------------------------
def _cfg(path: Optional[Path] = None) -> Dict:
    return json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


def combo(cfg: Dict) -> Dict:
    return {k: cfg.get(k) for k in CAMPOS_COMBO}


def config_motor(cfg: Dict) -> mm.Config:
    assert cfg.get("reglas") == "lotes" and cfg["ganancia"] == "G4", "el motor de lotes es G4"
    return ml.config(salida=cfg["salida"], comision=float(cfg.get("comision", mm.COMISION)),
                     slippage=float(cfg.get("slippage", mm.SLIPPAGE)),
                     min_orden=float(cfg.get("min_orden_usdt", mm.MIN_ORDEN_USDT)),
                     dia_cierre=int(cfg.get("dia_cierre", 3)))


def motor_nuevo(cfg: Dict) -> ml.MotorLotes:
    return ml.MotorLotes(config_motor(cfg), max_ciclos=int(cfg["ciclos_por_lote"]),
                         caida=float(cfg["caida_atrapado"]), dias=float(cfg["dias_atrapado"]))


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


class EstadoIlegible(Exception):
    pass


def _rutas(out: Optional[Path] = None):
    out = Path(out or OUT_DIR)
    return (out / "estado.json", out / "operaciones.csv", out / "mensual.csv", out / "equity.csv")


def leer_estado(out: Optional[Path] = None) -> Optional[Dict]:
    f = _rutas(out)[0]
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EstadoIlegible(str(exc)) from exc


def _guardar_estado(estado: Dict, out: Optional[Path] = None) -> None:
    f = _rutas(out)[0]
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


def describir_combo(cfg: Dict) -> str:
    """La combinación en palabras (los códigos tipo `E4_0.5` rompen el Markdown de Telegram)."""
    e, sa = rr.ENTRADAS[cfg["entrada"]], mm.SALIDAS[cfg["salida"]]
    slots = (f"lotes de {cfg['ciclos_por_lote']} ciclos, atrapado a −{_n(cfg['caida_atrapado'] * 100, 0)} % "
             f"o {cfg['dias_atrapado']} días")
    if e["tipo"] == "E4":
        compra = f"compra tras caer {_n(e['g'], 2).rstrip('0').rstrip(',')} × rango diario"
    else:
        compra = f"entrada {cfg['entrada']}"
    if sa["tipo"] == "S4":
        vende = f"vende en +{_n(sa['f'], 2).rstrip('0').rstrip(',')} × rango (mínimo +0,6 %)"
    else:
        vende = f"salida {cfg['salida']}"
    reinv = int(round(mm.GANANCIA[cfg["ganancia"]] * 100))
    return f"{slots} · {compra} · {vende} · {reinv} % de la ganancia queda en el bot"


def mensaje_operaciones_md(eventos: List[Dict]) -> str:
    lineas = ["_Modo papel: sin dinero real._", ""]
    for e in eventos:
        if e["tipo"] == "compra":
            lineas.append(f"🟢 *Compra* slot {e['slot']} · {_n(e['qty'], 6)} BTC a {_n(e['precio'], 0)} "
                          f"({_n(e['usdt'])} USDT) · vende en {_n(e['objetivo'], 0)} o más")
        elif e["tipo"] == "deposito":
            lineas.append(f"💵 Depósito del real: {_n(e['monto'])} USDT (lote {e['lote']})")
        else:
            lineas.append(f"🔴 *Venta* slot {e['slot']} a {_n(e['precio'], 0)} · ganancia "
                          f"{_signo(e['ganancia'])} USDT ({_signo(e['pct'], 2)} %) en {e['horas']} h")
    return "\n".join(lineas)


def mensaje_cierre_md(mes: Dict) -> str:
    return "\n".join([
        f"_Modo papel._ Corte del ciclo {mes['ciclo']} a {_n(mes['precio_cierre'], 0)} (sólo contable).",
        f"Ganancia realizada {_signo(mes['realizado'])} USDT · {int(mes['ventas'])} ventas · "
        f"{int(mes['entradas'])} compras",
        f"Atrapados reemplazados: {int(mes['reemplazos_deposito'] + mes['reemplazos_4'])} (costo "
        f"{_n(mes['atrapadas_costo'])} USDT) · pérdida latente {_signo(mes['latente'])} USDT",
        f"*Neto del ciclo {_signo(mes['neto'])} USDT* · depósitos {_n(mes['depositado'])} USDT",
    ])


def mensaje_resumen_md(cfg: Dict, m: ml.MotorLotes, precio: float, rango: float, ops: pd.DataFrame,
                       meses: pd.DataFrame, inicio: str) -> str:
    filas = ["```", "slot  estado     USDT   entrada  vende en"]
    for i, s in enumerate(m.slots, 1):
        if s.abierto:
            marca = "atrapado" if m.atrapado[i - 1] else "comprado"
            filas.append(f"{i:<5} {marca} {s.costo:7.1f} {s.entrada:9.0f} {m.precio_tp(s):9.0f}")
        elif s.nominal > 0 or i <= 3:
            filas.append(f"{i:<5} libre    {s.nominal:7.1f}")
    filas.append("```")
    gan = float(ops["ganancia"].sum()) if len(ops) else 0.0
    con_bot = m.patrimonio_btc(precio)
    return "\n".join([
        f"_Modo papel desde {inicio[:10]}, lotes de {_n(m.max_ciclos, 0)} ciclos._ BTC {_n(precio, 0)} · "
        f"rango diario {_n(rango * 100, 2)} %",
        *filas,
        f"Ventas: {len(ops)} · ganancia realizada {_signo(gan)} USDT · costo de atrapadas reemplazadas "
        f"{_n(m.st['atrapadas_costo'])} USDT · latente {_signo(m.latente(precio))} USDT",
        f"USDT del bot {_n(m.usdt_bot())} · holding {_n(m.holding_btc, 6)} BTC",
        f"BTC con bot {_n(con_bot, 6)} · DCA puro {_n(m.bench_btc, 6)} ({_signo((con_bot / m.bench_btc - 1) * 100, 2)} %)"
        if m.bench_btc else f"BTC con bot {_n(con_bot, 6)}",
    ])


def depositos_del_real(f: Path = None) -> pd.DataFrame:
    """Los depósitos que detectó el real (lote > 0; el 0 es su capital inicial). Si el archivo no
    se puede leer, falla: un depósito perdido rompería la sombra sin avisar."""
    f = f or DEPOSITOS_REAL
    if not f.exists():
        return pd.DataFrame(columns=["t", "lote", "monto"])
    d = pd.read_csv(f, parse_dates=["t"])
    d["t"] = d["t"].dt.tz_localize(None) if d["t"].dt.tz is not None else d["t"]
    d["lote"] = d["lote"].astype(int)
    return d[d["lote"] > 0]


def migrar_v04(estado: Dict, cfg: Dict, ahora: pd.Timestamp, out: Optional[Path] = None) -> Dict:
    """Pasa el estado v0.4 a lotes, con respaldo de la carpeta. Mismas reglas que el real."""
    import shutil
    out = Path(out or OUT_DIR)
    destino = out.parent / f"{out.name}_v04_{ahora:%Y%m%d%H%M}"
    shutil.copytree(out, destino)
    f_estado, f_ops, _, f_eq = _rutas(out)
    ops = pd.read_csv(f_ops) if f_ops.exists() else pd.DataFrame()
    eq_v04 = f_eq.with_name("equity_v04.csv")
    precio_inicio = float(pd.read_csv(f_eq if f_eq.exists() else eq_v04)["precio"].iloc[0])
    m = ml.migrar_desde_v04(estado["motor"], config_motor(cfg), pd.Timestamp(ahora.tz_localize(None)),
                            precio_inicio, float(estado["combo"]["capital"]), max_ciclos=int(cfg["ciclos_por_lote"]))
    # lo hecho con v0.4 entra en el primer corte, que va del arranque al primer día 3 después de migrar
    m.st["realizado"] = float(ops["ganancia"].sum()) if len(ops) else 0.0
    m.st["ventas"] = len(ops)
    m.st["entradas"] = len(ops) + sum(s["abierto"] for s in estado["motor"]["slots"])
    m.ciclo = mm.ciclo_de(pd.Timestamp(ahora.tz_localize(None)), int(cfg.get("dia_cierre", 3)))
    # las columnas de equity.csv cambian: el de v0.4 queda aparte y el nuevo empieza con su encabezado
    if f_eq.exists():
        os.replace(f_eq, eq_v04)
    log.warning("btc_rangos: estado v0.4 migrado a lotes (respaldo en %s)", destino)
    return dict(estado, combo=combo(cfg), reglas="lotes", inicio_lotes=str(ahora), respaldo_v04=str(destino),
                motor=m.a_dict(), precio_inicio=precio_inicio, lotes_aplicados=[])


# --- job ------------------------------------------------------------------------------
_AVISADO = set()


def run_btc_rangos(ahora: Optional[datetime] = None, fuentes: Optional[Dict[str, Callable]] = None,
                   config_path: Optional[Path] = None, out_dir: Optional[Path] = None) -> Optional[Dict]:
    """Job horario. Devuelve un resumen de la corrida, o None si no corrió. Nunca levanta.
    `config_path` y `out_dir` corren otra instancia (los experimentos de `btc_rangos_experimentos`)."""
    try:
        cfg = _cfg(config_path)
        out = Path(out_dir or OUT_DIR)
        info = _emitir if cfg.get("avisos", True) else (lambda *a, **k: None)  # los errores avisan igual
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
        out.mkdir(parents=True, exist_ok=True)
        f_estado, f_ops, f_mes, f_eq = _rutas(out)

        try:
            estado = leer_estado(out)
        except EstadoIlegible as exc:
            log.critical("btc_rangos: estado ilegible (%s); no se escribe nada", exc)
            _emitir("btc_rangos_estado_ilegible", "CRITICAL", detail=str(exc)[:300])
            return None

        v = velas_cerradas(velas_fn(VENTANA_VELAS), ahora)
        if len(v) < MIN_VELAS:
            log.warning("btc_rangos: %s velas cerradas (mínimo %s); se reintenta en 1 h", len(v), MIN_VELAS)
            return None
        datos = preparar(v, cfg)
        mcfg = config_motor(cfg)

        if estado is None:
            m = motor_nuevo(cfg)
            t0 = v.index[-1] + pd.Timedelta(hours=1)
            precio0 = float(v["close"].iloc[-1])
            capital = float(cfg["capital"])
            m.depositar(capital, 0, t0, precio0)
            m.sig_lote = 1
            m.iniciar_cortes(capital, precio0)
            m.ciclo = mm.ciclo_de(t0, mcfg.dia_cierre)
            estado = {"inicio": str(ahora), "inicio_lotes": str(ahora), "reglas": "lotes", "modo": modo,
                      "combo": combo(cfg), "ultima_barra": str(v.index[-1]), "ultimo_resumen": None,
                      "precio_inicio": precio0, "lotes_aplicados": [], "motor": m.a_dict()}
            _guardar_estado(estado, out)
            info("btc_rangos_inicio", "INFO", mensaje_md=(
                f"_Modo papel: sin dinero real._ Arranca con {_n(capital, 0)} USDT en 3 slots, lotes de "
                f"{cfg['ciclos_por_lote']} ciclos. {describir_combo(cfg)}. "
                f"Opera desde la vela que abre a las {t0:%H:%M} UTC."))
            return {"nuevo": True, "procesadas": 0}
        if estado.get("reglas") != "lotes" and cfg.get("reglas") == "lotes":
            estado = migrar_v04(estado, cfg, ahora, out)
            _guardar_estado(estado, out)
        if estado.get("combo") != combo(cfg):
            if f"combo:{out}" not in _AVISADO:
                _AVISADO.add(f"combo:{out}")
                _emitir("btc_rangos_modo_rechazado", "WARN",
                        detail="la config cambió respecto del estado; no se mezclan combinaciones. "
                               f"Para cambiar, archivar {out} y reiniciar.")
            return None

        m = ml.MotorLotes.desde_dict(estado["motor"], mcfg)
        deps = depositos_del_real(fuentes.get("depositos"))
        aplicados = set(estado.get("lotes_aplicados") or [])
        deps = deps[~deps["lote"].isin(aplicados) & (deps["t"] >= pd.Timestamp(estado["inicio_lotes"][:19]).floor("h"))]
        ultima = pd.Timestamp(estado["ultima_barra"])
        nuevas = datos[datos.index > ultima]
        if len(nuevas) and nuevas.index[0] > ultima + pd.Timedelta(hours=1):
            log.warning("btc_rangos: faltan velas entre %s y %s", ultima, nuevas.index[0])
        eventos, cierres = [], []
        for t, fila in nuevas.iterrows():
            if mm.ciclo_de(t, mcfg.dia_cierre) != m.ciclo:
                cierres.append(m.corte_mensual(t, float(fila["open"])))
                m.ciclo = mm.ciclo_de(t, mcfg.dia_cierre)
            llegan = deps[deps["t"] <= t]  # el real lo detectó en esta vela (o antes, si el papel iba atrasado)
            deps = deps[deps["t"] > t]
            a_entrar = list(m.pendientes)
            n_ops = len(m.operaciones)
            m.vela(t, fila["open"], fila["high"], fila["low"], fila["close"], bool(fila["senal"]),
                   float(fila["atr_prev"]), float(fila["rango_prev"]),
                   [(float(x.monto), int(x.lote)) for x in llegan.itertuples()])
            for x in llegan.itertuples():
                aplicados.add(int(x.lote))
                eventos.append({"tipo": "deposito", "monto": float(x.monto), "lote": int(x.lote)})
            for op in m.operaciones[n_ops:]:
                eventos.append({"tipo": "venta", "slot": op["slot"], "precio": op["salida"], "ganancia": op["ganancia"],
                                "pct": op["ganancia"] / op["costo"] * 100,
                                "horas": int((op["salida_t"] - op["entrada_t"]).total_seconds() // 3600)})
            for i in a_entrar:
                s = m.slots[i]
                op = next((o for o in m.operaciones[n_ops:] if o["slot"] == i + 1 and o["entrada_t"] == t), None)
                if s.desde != t and op is None:
                    continue  # no entró (sin capital suficiente)
                qty, entrada, usdt = (op["qty"], op["entrada"], op["costo"]) if op else (s.qty, s.entrada, s.costo)
                ref = mm.Slot(nominal=usdt, entrada=entrada)
                eventos.insert(len(eventos) - (1 if op else 0),
                               {"tipo": "compra", "slot": i + 1, "qty": qty, "precio": entrada, "usdt": usdt,
                                "objetivo": objetivo_venta(cfg, ref, float(fila["rango_prev"]))})
        _append_csv(f_ops, m.operaciones)
        _append_csv(f_mes, m.meses)
        m.operaciones, m.meses = [], []
        precio = float(v["close"].iloc[-1])
        rango_actual = float(datos["rango"].iloc[-1])
        _append_csv(f_eq, [{"ts": ahora, "precio": precio, "capital_bot": m.usdt_bot() + m.invertido(),
                            "usdt_bot": m.usdt_bot(), "btc_slots": m.btc_en_slots(), "holding_btc": m.holding_btc,
                            "bench_btc": m.bench_btc, "patrimonio_btc": m.patrimonio_btc(precio),
                            "latente": m.latente(precio), "atrapados": sum(m.atrapado)}])
        if len(nuevas):
            estado["ultima_barra"] = str(nuevas.index[-1])
        estado["motor"] = m.a_dict()
        estado["lotes_aplicados"] = sorted(aplicados)
        estado["actualizado"] = str(ahora)

        if eventos:
            info("btc_rangos_operaciones", "INFO", mensaje_md=mensaje_operaciones_md(eventos))
        for mes in cierres:
            info("btc_rangos_cierre_mensual", "INFO", mensaje_md=mensaje_cierre_md(mes))
        hoy = str(ahora.date())
        if estado.get("ultimo_resumen") != hoy and ahora.hour >= int(cfg.get("hora_resumen_utc", 12)):
            ops = pd.read_csv(f_ops) if f_ops.exists() else pd.DataFrame()
            meses = pd.read_csv(f_mes) if f_mes.exists() else pd.DataFrame()
            info("btc_rangos_resumen", "INFO",
                    mensaje_md=mensaje_resumen_md(cfg, m, precio, rango_actual, ops, meses, estado["inicio"]))
            estado["ultimo_resumen"] = hoy
        _guardar_estado(estado, out)
        return {"nuevo": False, "procesadas": len(nuevas), "eventos": len(eventos), "cierres": len(cierres)}
    except Exception as exc:  # el scheduler es de un solo hilo: nunca propagar
        log.exception("btc_rangos: error en la corrida: %s", exc)
        return None
