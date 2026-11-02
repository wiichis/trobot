"""Bot BTC por rangos en modo REAL (24/10/2026): manda órdenes a BingX spot.

Desde el 02/11/2026 opera con las reglas de lotes, "reponer sólo cuando hay dinero"
(`pkg/btc_rangos_lotes.py`, 3 ciclos por lote), con el mismo motor que el papel
(`pkg/btc_rangos.py`) y que la simulación (`scripts/btc_rangos_lotes_sim.py`). El usuario las
eligió aunque ninguna variante le gana al DCA puro en la simulación; los hitos para medirlo están
en `docs/btc_rangos_real_hitos.md`. El estado v0.4 (cierre forzado el día 3) se migra solo en la
primera corrida, con respaldo en `archivos/btc_rangos/real_v04_<fecha>/`.

Cómo ejecuta las reglas del motor en el mundo real, en el orden de la vela del motor:
- **Ventas**: apenas se compra, se deja en el exchange una orden LÍMITE de venta al TP
  (`Motor.precio_tp`: +0,75 × R, mínimo +0,6 %, nunca bajo el piso). Es el "si el máximo de la
  vela toca el TP, vende al TP" del backtest, y se ejecuta aunque el server esté caído. Cada
  hora se reemplaza si el TP cambió con el R vigente (en la práctica, una vez por día).
- **Atrapados**: al cierre de la vela, el slot con el close a −10 % de su entrada o con 30 días
  abierto. Mantiene su venta al TP hasta que lo reemplazan.
- **Compra de holding**: el capital de los tramos que cumplieron sus ciclos más el 10 % de la
  ganancia, a mercado, cuando suma la orden mínima.
- **Depósitos**: cada hora, el USDT de la cuenta que excede la contabilidad del bot más
  `usdt_ajeno` es un depósito (desde `min_deposito_usdt`, con tope `dca_maximo_usdt`; el exceso
  pasa a `usdt_ajeno`). Si cubre a un atrapado, se cancela la venta del atrapado y su BTC queda en
  la cuenta como holding (el bot no lo vende nunca). Registro en `depositos.csv`; el papel los lee
  de ahí para aplicarlos en la misma hora.
- **Compras**: la señal se calcula al cierre de la vela; el job corre al minuto :01 y compra a
  mercado (`quoteOrderQty` = USDT del slot). Es la "apertura de la vela siguiente" del backtest.
- **Día 3 (00:00 UTC)**: sólo un corte contable en `mensual.csv`. No opera.

La fuente de verdad es el exchange (lección del bot de 5 min): los fills salen del estado de la
orden (`executedQty`, `cummulativeQuoteQty`, `fee`, `feeAsset`), nunca de lo que el bot infiere.
Cada mutación se guarda enseguida (estado atómico) para que un corte a mitad de corrida se
reconcilie en la siguiente: una compra se anota ANTES de mandarse (`compra_en_curso`) y se busca
por su clientOrderID si no se llegó a registrar.

Falla cerrado:
- Estado ilegible → no hace nada y avisa.
- Una compra sin resolver, el USDT o el BTC de la cuenta por debajo de lo que el bot cree tener,
  una comisión en un activo desconocido, o el neto acumulado bajo `pausa_si_neto_menor_usdt`
  → PAUSA las compras (las ventas siguen) y avisa. La pausa la levanta una persona.

Estado y registros en `archivos/btc_rangos/real/`: `estado.json`, `operaciones.csv`,
`mensual.csv`, `depositos.csv`, `ordenes.csv` (cada orden con el libro al mandarla, para medir
el deslizamiento) y `equity.csv`.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import btc_rangos as papel
from . import btc_rangos_lotes as ml
from . import btc_rangos_motor as mm
from .bingx_spot import ClienteSpot, ErrorExchange, Orden, Reglas

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "btc_rangos_real.json"
OUT_DIR = papel.BASE_DIR / "archivos" / "btc_rangos" / "real"
TOLERANCIA_USDT = 1.0
TOLERANCIA_BTC = 2e-6
MIN_DEPOSITO_USDT = 15.0

# Neto del ciclo esperado por la simulación según cuánto se movió BTC (lotes de 3 ciclos, corte del
# día 3, 50 calendarios 2021-2026, `btc_rangos_lotes_sim.py --mensual 3`; medianas en % del capital
# del bot al empezar el ciclo).
ESPERADO_SIMULACION = ((-5.0, "cayó más de 5 %", -9.8), (5.0, "se movió menos de ±5 %", 1.1),
                       (math.inf, "subió más de 5 %", 9.6))


def esperado_simulacion(var_btc_pct: float):
    for tope, texto, neto in ESPERADO_SIMULACION:
        if var_btc_pct < tope:
            return texto, neto


class Pausa(Exception):
    pass


def _cfg() -> Dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _rutas() -> Dict[str, Path]:
    return {k: OUT_DIR / f for k, f in (("estado", "estado.json"), ("ops", "operaciones.csv"),
                                         ("mes", "mensual.csv"), ("ordenes", "ordenes.csv"), ("eq", "equity.csv"),
                                         ("dep", "depositos.csv"))}


def leer_estado() -> Optional[Dict]:
    f = _rutas()["estado"]
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as exc:
        raise papel.EstadoIlegible(str(exc)) from exc


def _emitir(categoria: str, severidad: str, **campos) -> None:
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event(categoria, severidad, force=True, **campos)  # en real no se pierde ningún aviso
    except Exception:
        pass


class Corrida:
    """Una corrida del job. Guarda el estado después de cada cambio que toca el exchange."""

    def __init__(self, cfg: Dict, estado: Dict, m: ml.MotorLotes, cli: ClienteSpot, reglas: Reglas,
                 ahora: pd.Timestamp):
        self.cfg, self.estado, self.m, self.cli, self.reglas, self.ahora = cfg, estado, m, cli, reglas, ahora
        self.symbol = cfg.get("symbol", "BTC-USDT")
        self.eventos: List[Dict] = []
        self.ordenes_log: List[Dict] = []
        self.depositos_log: List[Dict] = []
        # lo ya vendido de las ventas al TP que siguen vivas (llenas en parte): el USDT está en la
        # cuenta y el BTC ya no, pero el bot lo realiza recién cuando la orden termina
        self.parcial = {"usdt": 0.0, "btc": 0.0}

    # --- persistencia -------------------------------------------------------------
    def guardar(self, final: bool = False) -> None:
        """Operaciones, órdenes y depósitos van al CSV enseguida. El corte mensual espera en el
        estado hasta el final de la corrida."""
        r = _rutas()
        papel._append_csv(r["ops"], self.m.operaciones)
        papel._append_csv(r["ordenes"], self.ordenes_log)
        papel._append_csv(r["dep"], self.depositos_log)
        self.m.operaciones, self.ordenes_log, self.depositos_log = [], [], []
        if final:
            papel._append_csv(r["mes"], self.m.meses)
            self.m.meses = []
        self.estado["meses_pendientes"] = papel._a_json(self.m.meses)
        self.estado["motor"] = self.m.a_dict()
        self.estado["actualizado"] = str(self.ahora)
        tmp = r["estado"].with_suffix(".tmp")
        tmp.write_text(json.dumps(papel._a_json(self.estado), indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, r["estado"])

    def pausar(self, motivo: str) -> None:
        if not self.estado.get("pausa"):
            self.estado["pausa"] = {"motivo": motivo, "desde": str(self.ahora)}
            log.critical("btc_rangos_real: PAUSA de compras: %s", motivo)
            _emitir("btc_real_alerta", "CRITICAL", mensaje_md=(
                f"*Compras en pausa.* {motivo}\nLas ventas ya colocadas siguen activas. "
                "La pausa la levanta una persona después de revisar."))

    def en_pausa(self) -> bool:
        return bool(self.estado.get("pausa") or self.cfg.get("pausar_entradas"))

    def _registrar_orden(self, o: Orden, libro: Optional[Dict], slot: int, motivo: str) -> None:
        self.ordenes_log.append({"ts": self.ahora, "slot": slot, "motivo": motivo, "order_id": o.order_id,
                                 "client_id": o.client_id, "side": o.side, "tipo": o.tipo, "estado": o.estado,
                                 "precio": o.precio, "qty": o.qty, "ejecutado": o.ejecutado, "quote": o.quote,
                                 "precio_medio": o.precio_medio, "fee": o.fee, "fee_asset": o.fee_asset,
                                 "bid": (libro or {}).get("bid"), "ask": (libro or {}).get("ask")})

    # --- fills --------------------------------------------------------------------
    def _fee_usdt(self, o: Orden) -> float:
        if o.fee == 0 or o.fee_asset in ("USDT", ""):
            return o.fee
        if o.fee_asset == "BTC":
            return o.fee * o.precio_medio
        self.pausar(f"comisión en un activo desconocido ({o.fee_asset}) en la orden {o.order_id}")
        return o.quote * self.m.cfg.comision

    def _aplicar_venta(self, i: int, o: Orden, motivo: str) -> None:
        s = self.m.slots[i]
        neto = o.quote - self._fee_usdt(o)
        t = self.ahora.tz_localize(None).floor("h")
        if o.ejecutado + TOLERANCIA_BTC >= s.qty:
            horas = int((t - pd.Timestamp(s.desde)).total_seconds() // 3600)
            costo = s.costo
            self.m.cerrar_con_fill(i, t, neto, o.precio_medio, motivo)
            self.eventos.append({"tipo": "venta", "slot": i + 1, "precio": o.precio_medio, "ganancia": neto - costo,
                                 "pct": (neto - costo) / costo * 100, "horas": horas})
        else:
            costo = s.costo * o.ejecutado / s.qty
            self.m.venta_parcial(i, t, neto, o.ejecutado, o.precio_medio, motivo + "_parcial")
            self.eventos.append({"tipo": "venta_parcial", "slot": i + 1, "precio": o.precio_medio,
                                 "ganancia": neto - costo, "qty": o.ejecutado})

    def _aplicar_compra(self, i: int, o: Orden, t_ejec: pd.Timestamp) -> None:
        fee_btc = o.fee if o.fee_asset == "BTC" else 0.0
        costo = o.quote + (o.fee if o.fee_asset in ("USDT", "") else 0.0)
        if o.fee_asset not in ("BTC", "USDT", ""):
            self._fee_usdt(o)  # pausa
        qty_neta = o.ejecutado - fee_btc
        qty = float(self.reglas.qty_abajo(qty_neta))
        self.m.holding_btc += max(qty_neta - qty, 0.0)  # el polvo bajo el step no se puede vender: va a holding
        self.m.entrar_con_fill(i, t_ejec, costo, qty, o.precio_medio)
        self.eventos.append({"tipo": "compra", "slot": i + 1, "qty": qty, "precio": o.precio_medio, "usdt": costo,
                             "objetivo": self.m.precio_tp(self.m.slots[i])})

    def _aplicar_compra_holding(self, o: Orden) -> float:
        qty = o.ejecutado - (o.fee if o.fee_asset == "BTC" else 0.0)
        gastado = o.quote + (o.fee if o.fee_asset in ("USDT", "") else 0.0)
        self.m.holding_comprado(gastado, qty)
        self.eventos.append({"tipo": "holding", "qty": qty, "usdt": gastado, "precio": o.precio_medio})
        return qty

    # --- pasos --------------------------------------------------------------------
    def resolver_compra_en_curso(self) -> None:
        c = self.estado.get("compra_en_curso")
        if not c:
            return
        try:
            o = self.cli.consultar(self.symbol, order_id=c.get("order_id", ""), client_id=c["client_id"])
        except ErrorExchange as exc:
            if "code=100404" in str(exc):  # el exchange no la tiene: nunca se creó
                log.warning("btc_rangos_real: la compra %s no existe en el exchange; se descarta", c["client_id"])
                self.estado["compra_en_curso"] = None
                self.guardar()
                return
            self.pausar(f"no se pudo confirmar la compra {c['client_id']} del slot {c['slot']}: {exc}")
            return
        if not o.terminada:
            return
        self._registrar_orden(o, None, c["slot"], "compra_recuperada")
        if o.ejecutado > 0 and c.get("motivo") == "holding":
            self._aplicar_compra_holding(o)
        elif o.ejecutado > 0:  # la vela en que se compró (el estado guarda la hora UTC de la corrida)
            self._aplicar_compra(c["slot"] - 1, o, pd.Timestamp(c["t"]).tz_convert(None).floor("h"))
        self.estado["compra_en_curso"] = None
        self.guardar()

    def conciliar_ventas(self) -> None:
        # en orden de slot, como `Motor._salidas`: varias ventas en la misma vela se aplican igual
        for k, od in sorted(self.estado.get("ordenes", {}).items(), key=lambda x: int(x[0])):
            i = int(k) - 1
            o = self.cli.consultar(self.symbol, order_id=od["order_id"])
            if o.estado == "FILLED":
                self._registrar_orden(o, None, i + 1, "tp_lleno")
                self._aplicar_venta(i, o, "tp")
                del self.estado["ordenes"][k]
                self.guardar()
            elif o.terminada:  # cancelada por fuera del bot
                self._registrar_orden(o, None, i + 1, "tp_cancelada_afuera")
                if o.ejecutado > 0:
                    self._aplicar_venta(i, o, "tp")
                del self.estado["ordenes"][k]
                self.guardar()
                _emitir("btc_real_alerta", "WARN", mensaje_md=(
                    f"La venta del slot {i + 1} apareció cancelada ({o.estado}) sin que el bot la cancelara. "
                    "Se vuelve a colocar."))
            elif o.ejecutado > 0:
                self.parcial["usdt"] += o.quote - self._fee_usdt(o)
                self.parcial["btc"] += o.ejecutado

    def _cancelar(self, k: str, motivo: str) -> Orden:
        """Cancela y devuelve el estado final: puede haberse llenado (total o en parte) en el medio."""
        od = self.estado["ordenes"][k]
        try:
            self.cli.cancelar(self.symbol, od["order_id"])
        except ErrorExchange as exc:
            log.warning("btc_rangos_real: cancelar %s: %s (se consulta el estado)", od["order_id"], exc)
        o = self.cli.esperar_terminada(self.symbol, od["order_id"])
        self._registrar_orden(o, None, int(k), motivo)
        if not o.terminada:
            raise ErrorExchange(f"la orden {od['order_id']} sigue {o.estado} después de cancelarla")
        i = int(k) - 1
        if o.ejecutado > 0:
            self._aplicar_venta(i, o, "tp")
        del self.estado["ordenes"][k]
        self.guardar()
        return o

    def _liberar_atrapados(self, slots: List[int]) -> None:
        """Antes de reemplazar a un atrapado se cancela su venta: su BTC pasa a holding libre. Si
        la venta se llenó en el medio, el slot se cerró con ganancia y ya no hay que reemplazarlo;
        si se llenó en parte, esa parte se realiza ahora."""
        for i in slots:
            k = str(i + 1)
            if k in self.estado.get("ordenes", {}):
                o = self.cli.consultar(self.symbol, order_id=self.estado["ordenes"][k]["order_id"])
                if o.ejecutado > 0:   # deja de ser "parcial en curso": `_cancelar` lo realiza
                    self.parcial["usdt"] -= o.quote - self._fee_usdt(o)
                    self.parcial["btc"] -= o.ejecutado
                self._cancelar(k, "reemplazo_atrapado")

    def verificar_saldos(self) -> Dict[str, float]:
        sal = self.cli.saldos()
        usdt = sum(sal.get("USDT", {"free": 0, "locked": 0}).values())
        btc = sum(sal.get("BTC", {"free": 0, "locked": 0}).values())
        en_slots = sum(s.qty for s in self.m.slots if s.abierto) - self.parcial["btc"]
        if btc + TOLERANCIA_BTC < en_slots:
            self.pausar(f"la cuenta spot tiene {btc:.6f} BTC y los slots abiertos suman {en_slots:.6f}")
        return {"usdt": usdt, "usdt_libre": sal.get("USDT", {}).get("free", 0.0), "btc": btc}

    def corte_mensual(self, t_ejec: pd.Timestamp) -> None:
        """Día 3: sólo contabilidad. El ciclo se marca antes de guardar: un error después no lo repite."""
        fila = self.m.corte_mensual(t_ejec, self.cli.precio(self.symbol))
        self.m.ciclo = mm.ciclo_de(t_ejec, self.m.cfg.dia_cierre)
        self.eventos.append({"tipo": "cierre", "mes": fila})
        self.guardar()

    def comprar_holding(self, t_ejec: pd.Timestamp) -> None:
        monto = self.m.monto_holding()
        if monto <= 0 or self.en_pausa():
            return
        precio = self.cli.precio(self.symbol)
        minimo = max(self.m.cfg.min_orden, self.reglas.min_notional, self.reglas.min_qty * precio * 1.02)
        if monto < minimo:   # espera a sumar la orden mínima
            return
        o = self._comprar(monto, f"BRH{t_ejec:%y%m%d%H}", 0, "holding")
        if o.ejecutado > 0:
            self._aplicar_compra_holding(o)
        self.guardar()

    def detectar_deposito(self, t_ejec: pd.Timestamp, sal: Dict[str, float]) -> None:
        """El USDT que excede la contabilidad del bot más `usdt_ajeno` es un depósito, que entra
        como un lote nuevo. Lo que falta es plata que el bot cree tener y no está: pausa."""
        esperado = self.m.usdt_bot() + float(self.estado.get("usdt_ajeno", 0.0)) + self.parcial["usdt"]
        diferencia = sal["usdt"] - esperado
        if diferencia < -TOLERANCIA_USDT:
            self.pausar(f"hay {sal['usdt']:.2f} USDT en spot y el bot esperaba al menos {esperado:.2f}")
            return
        if diferencia < float(self.cfg.get("min_deposito_usdt", MIN_DEPOSITO_USDT)):
            return
        tope = float(self.cfg.get("dca_maximo_usdt", 500))
        monto = round(min(diferencia, tope), 2)
        self.estado["usdt_ajeno"] = float(self.estado.get("usdt_ajeno", 0.0)) + (diferencia - monto)
        precio = self.cli.precio(self.symbol)
        self._liberar_atrapados(self.m.reemplazaria(monto))
        antes = {i: self.m.slots[i].costo for i in range(3) if self.m.atrapado[i]}
        lote = self.m.sig_lote
        self.m.sig_lote += 1
        self.m.depositar(monto, lote, t_ejec, precio)
        reemplazados = [i + 1 for i in antes if not self.m.atrapado[i]]
        self.depositos_log.append({"ts": self.ahora, "t": t_ejec, "lote": lote, "monto": monto,
                                   "excedente": diferencia - monto, "precio": precio,
                                   "reemplaza": " ".join(map(str, reemplazados))})
        self.eventos.append({"tipo": "deposito", "monto": monto, "lote": lote, "reemplaza": reemplazados,
                             "ajeno": diferencia - monto})
        self.guardar()

    def chequear_cuarto(self, t_ejec: pd.Timestamp) -> None:
        a_reemplazar = self.m.reemplazaria(0.0)
        precio = float("nan")
        if a_reemplazar:
            self._liberar_atrapados(a_reemplazar)
            precio = self.cli.precio(self.symbol)
        antes = [i for i in range(3) if self.m.atrapado[i]]
        self.m.chequear_cuarto(t_ejec, precio)
        reemplazados = [i + 1 for i in antes if not self.m.atrapado[i] and not self.m.slots[i].abierto]
        if reemplazados:
            self.eventos.append({"tipo": "reemplazo_4", "reemplaza": reemplazados})
        self.guardar()

    def _comprar(self, usdt: float, client_id: str, slot: int, motivo: str) -> Orden:
        libro = self.cli.libro(self.symbol)
        monto = f"{math.floor(usdt * 100) / 100:.2f}"
        self.estado["compra_en_curso"] = {"slot": slot, "client_id": client_id, "t": str(self.ahora), "usdt": monto,
                                          "motivo": motivo}
        self.guardar()
        try:
            o = self.cli.comprar_mercado(self.symbol, monto, client_id)
        except ErrorExchange:
            self.estado["compra_en_curso"] = None   # respuesta definitiva: la orden no se creó
            self.guardar()
            raise
        self.estado["compra_en_curso"]["order_id"] = o.order_id
        self.guardar()
        o = self.cli.esperar_terminada(self.symbol, o.order_id)
        self._registrar_orden(o, libro, slot, motivo)
        if not o.terminada:
            raise ErrorExchange(f"la compra {o.order_id} sigue {o.estado}")
        self.estado["compra_en_curso"] = None
        return o

    def actualizar_ventas(self) -> None:
        for i, s in enumerate(self.m.slots):
            if not s.abierto:
                continue
            k = str(i + 1)
            precio = self.reglas.precio_arriba(self.m.precio_tp(s))
            od = self.estado.setdefault("ordenes", {}).get(k)
            if od and abs(float(od["precio"]) - float(precio)) < self.reglas.tick / 2:
                continue
            if od:
                o = self.cli.consultar(self.symbol, order_id=od["order_id"])
                if o.ejecutado > 0:
                    continue  # llenándose: no se toca hasta que termine
                o = self._cancelar(k, "tp_reemplazo")
                if not s.abierto:
                    continue  # se vendió justo
            qty = self.reglas.qty_abajo(s.qty)
            if float(qty) < self.reglas.min_qty:
                self.pausar(f"el slot {k} tiene {s.qty:.6f} BTC, menos que la orden mínima de venta")
                continue
            libro = self.cli.libro(self.symbol)
            cid = f"BRS{k}{self.ahora:%y%m%d%H%M}"
            o = self.cli.vender_limite(self.symbol, qty, precio, cid)
            self.estado["ordenes"][k] = {"order_id": o.order_id, "client_id": cid, "precio": precio, "qty": qty,
                                         "desde": str(self.ahora)}
            self._registrar_orden(o, libro, i + 1, "tp_colocada")
            self.guardar()

    def comprar_senal(self, t_ejec: pd.Timestamp, sal: Dict[str, float]) -> None:
        if self.en_pausa():
            self.m.pendientes = []
            return
        for i in list(self.m.pendientes):
            s = self.m.slots[i]
            self.m._sync(i)
            minimo = max(self.m.cfg.min_orden, self.reglas.min_notional)
            if s.abierto or s.nominal < minimo:
                continue
            if sal["usdt_libre"] + TOLERANCIA_USDT < s.nominal:
                self.pausar(f"para comprar el slot {i + 1} hacen falta {s.nominal:.2f} USDT y hay "
                            f"{sal['usdt_libre']:.2f} libres en spot")
                break
            o = self._comprar(min(s.nominal, sal["usdt_libre"]), f"BRB{i + 1}{t_ejec:%y%m%d%H}", i + 1, "compra")
            if o.ejecutado > 0:
                self._aplicar_compra(i, o, t_ejec)
                sal["usdt_libre"] -= o.quote
            self.guardar()
        self.m.pendientes = []


# --- mensajes -------------------------------------------------------------------------
_n, _signo = papel._n, papel._signo
MARCA = "*Dinero real.*"


def mensaje_operaciones_md(eventos: List[Dict]) -> str:
    lineas = [MARCA, ""]
    for e in eventos:
        if e["tipo"] == "compra":
            lineas.append(f"🟢 *Compra* slot {e['slot']} · {_n(e['qty'], 6)} BTC a {_n(e['precio'], 0)} "
                          f"({_n(e['usdt'])} USDT) · orden de venta en {_n(e['objetivo'], 0)}")
        elif e["tipo"] == "venta":
            lineas.append(f"🔴 *Venta* slot {e['slot']} a {_n(e['precio'], 0)} · ganancia "
                          f"{_signo(e['ganancia'])} USDT ({_signo(e['pct'], 2)} %) en {e['horas']} h")
        elif e["tipo"] == "venta_parcial":
            lineas.append(f"🔴 *Venta parcial* slot {e['slot']}: {_n(e['qty'], 6)} BTC a {_n(e['precio'], 0)} · "
                          f"ganancia {_signo(e['ganancia'])} USDT")
        elif e["tipo"] == "deposito":
            txt = f"💵 *Depósito* {_n(e['monto'])} USDT detectado (lote {e['lote']})"
            if e["reemplaza"]:
                txt += f" · reemplaza a los atrapados {', '.join(map(str, e['reemplaza']))}, que pasan a holding"
            if e["ajeno"] > 0.005:
                txt += f" · {_n(e['ajeno'])} USDT sobre el tope quedan aparte"
            lineas.append(txt)
        elif e["tipo"] == "reemplazo_4":
            lineas.append(f"🔁 El 4.º slot reemplaza a los atrapados {', '.join(map(str, e['reemplaza']))}, "
                          "que pasan a holding")
        elif e["tipo"] == "holding":
            lineas.append(f"🟠 *Holding*: {_n(e['qty'], 6)} BTC a {_n(e['precio'], 0)} ({_n(e['usdt'])} USDT)")
    return "\n".join(lineas)


def mensaje_cierre_md(mes: Dict, var_btc: Optional[float], acumulado: float) -> str:
    cap = mes["capital_inicio"]
    pct = mes["neto"] / cap * 100 if cap else 0.0
    lineas = [
        f"{MARCA} Corte del ciclo {mes['ciclo']} a {_n(mes['precio_cierre'], 0)} (sólo contable, no opera).",
        f"Ganancia realizada {_signo(mes['realizado'])} USDT · {int(mes['ventas'])} ventas · "
        f"{int(mes['entradas'])} compras",
        f"Atrapados reemplazados: {int(mes['reemplazos_deposito'] + mes['reemplazos_4'])} (costo "
        f"{_n(mes['atrapadas_costo'])} USDT) · pérdida latente de lo abierto {_signo(mes['latente'])} USDT",
        f"*Neto del ciclo {_signo(mes['neto'])} USDT ({_signo(pct, 1)} %)* · acumulado {_signo(acumulado)} USDT",
        f"Depósitos {_n(mes['depositado'])} USDT · holding {_n(mes['holding_btc'], 6)} BTC",
    ]
    if var_btc is not None:
        texto, esperado = esperado_simulacion(var_btc)
        lineas.append(f"BTC {_signo(var_btc, 1)} % en el ciclo ({texto}): para eso la simulación esperaba un neto de "
                      f"~{_signo(esperado, 1)} %")
    return "\n".join(lineas)


def mensaje_resumen_md(cfg: Dict, est: Dict, m: ml.MotorLotes, precio: float, rango: float, ops: pd.DataFrame,
                       meses: pd.DataFrame) -> str:
    filas = ["```", "slot  estado     USDT   entrada  venta en"]
    for i, s in enumerate(m.slots, 1):
        if s.abierto:
            od = est.get("ordenes", {}).get(str(i))
            venta = f"{float(od['precio']):9.0f}" if od else "   sin orden"
            marca = "atrapado" if m.atrapado[i - 1] else "comprado"
            filas.append(f"{i:<5} {marca} {s.costo:7.1f} {s.entrada:9.0f} {venta}")
        elif s.nominal > 0 or i <= 3:
            filas.append(f"{i:<5} libre    {s.nominal:7.1f}")
    filas.append("```")
    gan = float(ops["ganancia"].sum()) if len(ops) else 0.0
    con_bot = m.patrimonio_btc(precio)
    lineas = [
        f"{MARCA} Desde {est['inicio'][:10]}, lotes de {_n(m.max_ciclos, 0)} ciclos desde {est['inicio_lotes'][:10]}. "
        f"BTC {_n(precio, 0)} · rango diario {_n(rango * 100, 2)} %",
        *filas,
        f"Ventas: {len(ops)} · ganancia realizada {_signo(gan)} USDT · costo de atrapadas reemplazadas "
        f"{_n(m.st['atrapadas_costo'])} USDT",
        f"Pérdida latente de lo abierto {_signo(m.latente(precio))} USDT",
        f"USDT del bot {_n(m.usdt_bot())} · para holding {_n(m.monto_holding())} · holding {_n(m.holding_btc, 6)} BTC",
        f"BTC con bot {_n(con_bot, 6)} · DCA puro {_n(m.bench_btc, 6)} ({_signo((con_bot / m.bench_btc - 1) * 100, 2)} %)"
        if m.bench_btc else f"BTC con bot {_n(con_bot, 6)}",
    ]
    if est.get("pausa"):
        lineas.append(f"⏸️ Compras en pausa desde {est['pausa']['desde'][:16]}: {est['pausa']['motivo']}")
    return "\n".join(lineas)


# --- estado ---------------------------------------------------------------------------
def migrar_v04(estado: Dict, cfg: Dict, combo: Dict, ahora: pd.Timestamp) -> Dict:
    """Pasa el estado v0.4 a lotes, sin tocar el exchange: las órdenes de venta vivas siguen.
    Respalda la carpeta entera antes. El DCA puro compra el capital inicial al precio del arranque."""
    import shutil
    destino = OUT_DIR.parent / f"real_v04_{ahora:%Y%m%d%H%M}"
    shutil.copytree(OUT_DIR, destino)
    r = _rutas()
    ops = pd.read_csv(r["ops"]) if r["ops"].exists() else pd.DataFrame()
    eq_v04 = r["eq"].with_name("equity_v04.csv")
    precio_inicio = float(estado.get("precio_cierre_anterior")
                          or pd.read_csv(r["eq"] if r["eq"].exists() else eq_v04)["precio"].iloc[0])
    m = ml.migrar_desde_v04(estado["motor"], papel.config_motor(cfg), pd.Timestamp(ahora.tz_localize(None)),
                            precio_inicio, float(estado["combo"]["capital"]),
                            max_ciclos=int(cfg["ciclos_por_lote"]))
    # lo hecho con v0.4 entra en el primer corte, que va del arranque al primer día 3 después de migrar
    m.st["realizado"] = float(ops["ganancia"].sum()) if len(ops) else 0.0
    m.st["ventas"] = len(ops)
    m.st["entradas"] = len(ops) + sum(s["abierto"] for s in estado["motor"]["slots"])
    m.ciclo = mm.ciclo_de(pd.Timestamp(ahora.tz_localize(None)), int(cfg.get("dia_cierre", 3)))
    nuevo = dict(estado, combo=combo, reglas="lotes", inicio_lotes=str(ahora), respaldo_v04=str(destino),
                 motor=m.a_dict(), precio_inicio=precio_inicio)
    nuevo.pop("precio_cierre_anterior", None)
    nuevo.pop("ultimo_recordatorio", None)
    # las columnas de equity.csv cambian: el de v0.4 queda aparte y el nuevo empieza con su encabezado
    if r["eq"].exists():
        os.replace(r["eq"], eq_v04)
    log.warning("btc_rangos_real: estado v0.4 migrado a lotes (respaldo en %s)", destino)
    return nuevo


# --- job ------------------------------------------------------------------------------
_AVISADO = set()
_ERRORES = {"seguidos": 0}


def run_btc_rangos_real(ahora: Optional[datetime] = None, cliente: Optional[ClienteSpot] = None,
                        velas: Optional[Callable[[int], pd.DataFrame]] = None) -> Optional[Dict]:
    """Job horario (:01). Devuelve un resumen, o None si no corrió. Nunca levanta."""
    corrida = None
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return None
        if cfg.get("modo") != "real":
            if "modo" not in _AVISADO:
                _AVISADO.add("modo")
                _emitir("btc_real_alerta", "WARN", detail=f"btc_rangos_real.json tiene modo={cfg.get('modo')!r}; no corre")
            return None
        ahora = pd.Timestamp(ahora or datetime.now(timezone.utc))
        ahora = ahora.tz_localize("UTC") if ahora.tzinfo is None else ahora.tz_convert("UTC")
        symbol = cfg.get("symbol", "BTC-USDT")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            estado = leer_estado()
        except papel.EstadoIlegible as exc:
            log.critical("btc_rangos_real: estado ilegible (%s); no se hace nada", exc)
            _emitir("btc_real_alerta", "CRITICAL", mensaje_md=f"*Estado ilegible.* No se opera hasta revisarlo. {exc}"[:500])
            return None
        cli = cliente or ClienteSpot.desde_credenciales()
        reglas = cli.reglas(symbol)
        if not (reglas.compra_api and reglas.venta_api):
            log.warning("btc_rangos_real: el par no acepta órdenes por API ahora (compra=%s venta=%s)",
                        reglas.compra_api, reglas.venta_api)
            return None
        velas_fn = velas or (lambda limit: papel._velas_bingx(symbol, limit))
        v = papel.velas_cerradas(velas_fn(papel.VENTANA_VELAS), ahora)
        if len(v) < papel.MIN_VELAS:
            log.warning("btc_rangos_real: %s velas cerradas; se reintenta en 1 h", len(v))
            return None
        datos = papel.preparar(v, cfg)
        combo = papel.combo(cfg)
        mcfg = papel.config_motor(cfg)
        t_barra = v.index[-1]
        t_ejec = t_barra + pd.Timedelta(hours=1)

        if estado is None:
            sal = cli.saldos()
            usdt = sum(sal.get("USDT", {"free": 0, "locked": 0}).values())
            libre = sal.get("USDT", {}).get("free", 0.0)
            capital = float(cfg["capital"])
            if libre + 1e-9 < capital:
                if "capital" not in _AVISADO:
                    _AVISADO.add("capital")
                    _emitir("btc_real_alerta", "WARN", mensaje_md=(
                        f"{MARCA} No arranca: la cuenta spot tiene {_n(libre)} USDT libres y el bot necesita "
                        f"{_n(capital, 0)}. Transfiere desde futuros a spot y arranca solo en la próxima hora."))
                return None
            com = cli.comisiones(symbol)
            m = papel.motor_nuevo(cfg)
            m.ciclo = mm.ciclo_de(t_ejec, mcfg.dia_cierre)
            precio = cli.precio(symbol)
            m.depositar(capital, 0, t_ejec, precio)
            m.sig_lote = 1
            m.iniciar_cortes(capital, precio)
            estado = {"inicio": str(ahora), "inicio_lotes": str(ahora), "reglas": "lotes", "modo": "real",
                      "combo": combo, "ultima_barra": str(t_barra), "ultimo_resumen": None, "ordenes": {},
                      "compra_en_curso": None, "pausa": None, "usdt_ajeno": usdt - capital,
                      "comisiones_cuenta": com, "precio_inicio": precio}
            corrida = Corrida(cfg, estado, m, cli, reglas, ahora)
            corrida.depositos_log.append({"ts": ahora, "t": t_ejec, "lote": 0, "monto": capital, "excedente": 0.0,
                                          "precio": precio, "reemplaza": ""})
            corrida.guardar()
            aviso = ""
            if max(com.values()) > mcfg.comision + 1e-9:
                aviso = f" ⚠️ La comisión de la cuenta ({com}) es mayor que la del backtest ({mcfg.comision})."
            _emitir("btc_real_inicio", "INFO", mensaje_md=(
                f"{MARCA} Arranca con {_n(capital, 0)} USDT en 3 slots, lotes de {cfg['ciclos_por_lote']} ciclos. "
                f"{papel.describir_combo(cfg)}. Primera decisión con la vela que cierra a las {t_ejec:%H:%M} UTC.{aviso}"))
            return {"nuevo": True}
        if estado.get("reglas") != "lotes" and cfg.get("reglas") == "lotes":
            estado = migrar_v04(estado, cfg, combo, ahora)
            _emitir("btc_real_inicio", "INFO", mensaje_md=(
                f"{MARCA} Pasa a lotes de {cfg['ciclos_por_lote']} ciclos, sin corte el día 3: un slot atrapado "
                "espera con su venta y se reemplaza cuando depositas USDT en spot. Las órdenes vivas siguen."))
        if estado.get("combo") != combo:
            if "combo" not in _AVISADO:
                _AVISADO.add("combo")
                _emitir("btc_real_alerta", "WARN", detail="la config real cambió respecto del estado; no corre")
            return None

        m = ml.MotorLotes.desde_dict(estado["motor"], mcfg)
        m.meses = list(estado.get("meses_pendientes") or [])
        corrida = Corrida(cfg, estado, m, cli, reglas, ahora)
        corrida.guardar()  # una migración recién hecha queda escrita antes de tocar el exchange
        corrida.resolver_compra_en_curso()
        corrida.conciliar_ventas()
        # Mismo orden que `MotorLotes.vela`: al cierre de la vela, atrapados y señal; después, en la
        # vela que abre, el corte contable, la compra de holding, los depósitos, el 4.º slot y las
        # compras.
        m.pendientes = []
        nueva = t_barra > pd.Timestamp(estado["ultima_barra"])
        if nueva:
            for s in m.slots:
                s.entro_en_vela = False
            c = float(v["close"].iloc[-1])
            m.marcar_atrapados(t_ejec, c)
            m.rango = float(datos["rango_prev"].iloc[-1])
            m._senales(t_barra, c, bool(datos["senal"].iloc[-1]))
        if mm.ciclo_de(t_ejec, mcfg.dia_cierre) != m.ciclo:
            corrida.corte_mensual(t_ejec)
        corrida.comprar_holding(t_ejec)
        sal = corrida.verificar_saldos()
        corrida.detectar_deposito(t_ejec, sal)
        corrida.chequear_cuarto(t_ejec)
        sal = corrida.verificar_saldos()
        m.rango = float(datos["rango"].iloc[-1])   # el R con el que opera la vela que abre ahora
        corrida.actualizar_ventas()
        if m.pendientes:
            corrida.comprar_senal(t_ejec, sal)
            corrida.actualizar_ventas()
        if nueva:
            estado["ultima_barra"] = str(t_barra)

        # control de pérdida y avisos
        r = _rutas()
        nuevos_meses = [e["mes"] for e in corrida.eventos if e["tipo"] == "cierre"]
        precio = float(v["close"].iloc[-1])
        sal = corrida.verificar_saldos()
        papel._append_csv(r["eq"], [{"ts": ahora, "precio": precio, "capital_bot": m.usdt_bot() + m.invertido(),
                                     "usdt_bot": m.usdt_bot(), "btc_slots": m.btc_en_slots(),
                                     "holding_btc": m.holding_btc, "bench_btc": m.bench_btc,
                                     "patrimonio_btc": m.patrimonio_btc(precio), "usdt_cuenta": sal["usdt"],
                                     "btc_cuenta": sal["btc"], "latente": m.latente(precio),
                                     "atrapados": sum(m.atrapado)}])
        corrida.guardar(final=True)
        meses = pd.read_csv(r["mes"]) if r["mes"].exists() else pd.DataFrame()
        acumulado = float(meses["neto"].sum()) if len(meses) and "neto" in meses else 0.0
        limite = cfg.get("pausa_si_neto_menor_usdt")
        if nuevos_meses and limite is not None and acumulado < float(limite):
            corrida.pausar(f"el neto acumulado ({_signo(acumulado)} USDT) quedó bajo el límite de {_n(float(limite), 0)}")
            corrida.guardar()

        ops_eventos = [e for e in corrida.eventos if e["tipo"] != "cierre"]
        if ops_eventos:
            _emitir("btc_real_operaciones", "INFO", mensaje_md=mensaje_operaciones_md(ops_eventos))
        for mes in nuevos_meses:
            var = (mes["precio_cierre"] / mes["precio_inicio"] - 1) * 100
            _emitir("btc_real_cierre_mensual", "INFO", mensaje_md=mensaje_cierre_md(mes, var, acumulado))
        hoy = str(ahora.date())
        hora = int(cfg.get("hora_resumen_utc", 12))
        if estado.get("ultimo_resumen") != hoy and ahora.hour >= hora:
            ops = pd.read_csv(r["ops"]) if r["ops"].exists() else pd.DataFrame()
            _emitir("btc_real_resumen", "INFO", mensaje_md=mensaje_resumen_md(
                cfg, estado, m, precio, float(datos["rango"].iloc[-1]), ops, meses))
            estado["ultimo_resumen"] = hoy
        corrida.guardar()
        _ERRORES["seguidos"] = 0
        return {"nuevo": False, "eventos": len(corrida.eventos), "pausa": bool(estado.get("pausa"))}
    except Exception as exc:  # el scheduler es de un solo hilo: nunca propagar
        log.exception("btc_rangos_real: error en la corrida: %s", exc)
        try:
            if corrida is not None:
                corrida.guardar()
        except Exception:
            log.exception("btc_rangos_real: no se pudo guardar el estado después del error")
        _ERRORES["seguidos"] += 1
        if _ERRORES["seguidos"] % 6 == 1:
            _emitir("btc_real_alerta", "CRITICAL", mensaje_md=(
                f"*Error en la corrida* ({_ERRORES['seguidos']} seguidas). {type(exc).__name__}: {exc}")[:600])
        return None
