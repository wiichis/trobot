"""Motor del bot BTC por rangos con lotes: "reponer sólo cuando hay dinero" (02/11/2026).

Es el mismo motor de `scripts/btc_rangos_lotes_sim.py` (27-28/10, informe en
`docs/btc_rangos_lotes_informe.md`), movido acá para que la simulación, el papel y el real usen
el mismo código. Lo eligió el usuario el 02/11 con 3 ciclos por lote, aunque ninguna variante
le gana al DCA puro en la simulación (1 ciclo −0,65 %, 4 ciclos −1,73 %).

Reglas (fijadas el 27/10, variante B):
- Cada depósito de USDT es un lote que se reparte en tercios entre los slots 1-3. Cada tramo
  hace como máximo `max_ciclos` ciclos (compra → venta al TP) y después su capital compra BTC
  de holding. La ganancia es del bot: 90 % se reinvierte en tercios (tramos sin límite de
  ciclos) y 10 % compra BTC de holding (G4).
- No hay corte el día 3. Al cierre de una vela, un slot abierto queda atrapado si el close está
  a −10 % de su entrada o lleva 30 días abierto. Es pegajoso (sale sólo si vende o si lo
  reemplazan) y mantiene su TP.
- Un depósito reemplaza al atrapado más barato que cubra por sí solo: el BTC del atrapado pasa
  a holding (costo de atrapada = costo − qty × precio) y el slot recibe el USDT nuevo. Lo que no
  cubre a ningún atrapado va a un 4.º slot temporal, que reemplaza a un atrapado cuando su USDT
  lo cubre aunque esté abierto (`espera_cuenta`). Sin atrapados, el depósito va en tercios (si
  un slot está abierto, su tercio espera y entra con la próxima entrada).
- Entrada E4_0.5, salida S4_0.75 con piso de costo y 1 entrada por día, como `Motor`.

Orden dentro de la vela (`vela`), al precio de apertura: compra de holding pendiente →
depósitos → 4.º slot → entradas decididas en la vela anterior → salidas intrabarra → atrapados
al cierre → señales para la vela siguiente.

El día 3 queda sólo como corte contable (`corte_mensual`): no opera, no atrapa y no compra.
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

if __package__:
    from . import btc_rangos_motor as mm
else:  # cargado por ruta desde un script de laboratorio (importar `pkg` arrastra credenciales)
    mm = sys.modules["btc_rangos_motor"]

MAX_CICLOS = 3
CAIDA_ATRAPADO = 0.10
DIAS_ATRAPADO = 30
REINV = mm.GANANCIA["G4"]
COSTO_COMPRA = (1 + mm.SLIPPAGE) / (1 - mm.COMISION)  # precio efectivo por BTC = precio × COSTO_COMPRA
PERM = -1  # lote de los tramos de ganancia reinvertida: no tienen límite de ciclos
N_SLOTS = 4  # 3 fijos y el 4.º temporal


def config(salida: str = "S4_0.75", comision: float = mm.COMISION, slippage: float = mm.SLIPPAGE,
           min_orden: float = mm.MIN_ORDEN_USDT, dia_cierre: int = 3) -> mm.Config:
    return mm.Config(slots="A", salida=salida, ganancia="G4", capital=0.0, n_slots=N_SLOTS, cierre_forzado=False,
                     comision=comision, slippage=slippage, min_orden=min_orden, dia_cierre=dia_cierre)


class MotorLotes(mm.Motor):
    """Slots 0-2 fijos y el 3 temporal. Cada slot guarda tramos {lote, usdt, ciclos, t0, p0}."""

    def __init__(self, cfg: mm.Config, max_ciclos: float = MAX_CICLOS, atrapar: bool = True,
                 espera_cuenta: bool = True, caida: float = CAIDA_ATRAPADO, dias: float = DIAS_ATRAPADO):
        super().__init__(cfg=cfg, dca=lambda k: 0.0)
        self.max_ciclos, self.atrapar, self.espera_cuenta = max_ciclos, atrapar, espera_cuenta
        self.caida, self.dias = caida, dias
        self.tr = [[] for _ in range(N_SLOTS)]
        self.esp = [[] for _ in range(N_SLOTS)]
        self.atrapado = [False] * N_SLOTS
        self.pend_holding = 0.0
        self.reserva = 0.0
        self.reserva_filtro: list = []  # sólo el laboratorio: depósitos que esperan la ventana COMPRA
        self.bench_btc = 0.0
        self.st = {"atr_usdt": 0.0, "atr_btc_bot": 0.0, "atr_btc_dca": 0.0, "depositado": 0.0, "realizado": 0.0,
                   "atrapadas_costo": 0.0, "n_atrapadas": 0, "reemplazos_deposito": 0, "reemplazos_4": 0,
                   "holding_gastado": 0.0, "base_atrapadas": 0.0, "usdt_sale_ciclos": 0.0,
                   "usdt_sale_atrapado": 0.0, "ocioso": 0.0, "total": 0.0, "ventas_4": 0, "ventas": 0,
                   "entradas": 0}
        self.dias_4_ciclos: list = []
        self.corte_previo: Dict = {}
        self.sig_lote = 0

    # --- tramos -------------------------------------------------------------------
    def _sync(self, i: int):
        if not self.slots[i].abierto:
            self.slots[i].nominal = sum(x["usdt"] for x in self.tr[i])

    def _add(self, i: int, tramo: dict):
        dest = self.esp[i] if self.slots[i].abierto else self.tr[i]
        if tramo["lote"] == PERM:
            for x in dest:
                if x["lote"] == PERM:
                    x["usdt"] += tramo["usdt"]
                    self._sync(i)
                    return
        dest.append(tramo)
        self._sync(i)

    def _tercios(self, tramo: dict):
        for k in range(3):
            self._add(k, {**tramo, "usdt": tramo["usdt"] / 3})

    def _atrapados(self):
        return sorted((i for i in range(3) if self.atrapado[i]), key=lambda i: self.slots[i].costo)

    def _reemplazar(self, i: int, nuevos: list, t: pd.Timestamp, precio: float, via: str):
        s = self.slots[i]
        for x in self.tr[i]:
            if x["lote"] != PERM:  # el dinero de depósitos atrapado, contra comprarlo el día que llegó
                self.st["atr_usdt"] += x["usdt"]
                self.st["atr_btc_bot"] += x["usdt"] / s.costo * s.qty
                self.st["atr_btc_dca"] += x["usdt"] / (x["p0"] * COSTO_COMPRA)
        self.holding_btc += s.qty
        self.st["atrapadas_costo"] += s.costo - s.qty * precio
        self.st["base_atrapadas"] += s.costo
        self.st["usdt_sale_atrapado"] += sum(x["usdt"] for x in self.tr[i] if x["lote"] != PERM)
        self.st[f"reemplazos_{via}"] += 1
        # en el real la compra gasta algo menos que sus tramos (centavos, fill): eso sigue en USDT
        sobra = sum(x["usdt"] for x in self.tr[i]) - s.costo
        s.abierto, s.qty, s.costo, s.armado = False, 0.0, 0.0, False
        self.atrapado[i] = False
        self.tr[i], viejos = [], self.esp[i]
        self.esp[i] = []
        for x in nuevos + viejos:
            self._add(i, x)
        if sobra > 1e-6:
            self._tercios({"lote": PERM, "usdt": sobra, "ciclos": 0, "t0": t, "p0": precio})

    # --- pasos --------------------------------------------------------------------
    def registrar(self, usdt: float, precio: float):
        """El depósito llega: cuenta como depositado y el DCA puro lo compra a este precio."""
        self.st["depositado"] += usdt
        self.bench_btc += self._comprar_bench(usdt, precio)[0]

    def depositar(self, usdt: float, lote: int, t: pd.Timestamp, precio: float):
        self.registrar(usdt, precio)
        self.asignar(usdt, lote, t, precio, t, precio)

    def asignar(self, usdt: float, lote: int, t: pd.Timestamp, precio: float, t0: pd.Timestamp, p0: float):
        """El USDT entra al bot ahora (`t`, `precio`); `t0` y `p0` son los del día del depósito."""
        resto = usdt
        for i in self._atrapados():
            costo = self.slots[i].costo
            if resto + 1e-9 >= costo:
                self._reemplazar(i, [{"lote": lote, "usdt": costo, "ciclos": 0, "t0": t0, "p0": p0}], t, precio,
                                 "deposito")
                resto -= costo
        if resto <= 1e-9:
            return
        tramo = {"lote": lote, "usdt": resto, "ciclos": 0, "t0": t0, "p0": p0}
        if self._atrapados():
            self._add(3, tramo)
        else:
            self._tercios(tramo)

    def caja_cuarto(self) -> float:
        """USDT del 4.º slot que puede reemplazar a un atrapado ahora."""
        if self.slots[3].abierto:
            return sum(x["usdt"] for x in self.esp[3]) if self.espera_cuenta else 0.0
        return sum(x["usdt"] for x in self.tr[3])

    def reemplazaria(self, usdt: float = 0.0) -> List[int]:
        """Atrapados que un depósito de `usdt`, o el USDT del 4.º slot, alcanzan a reemplazar. El
        real cancela antes sus ventas al TP: el BTC tiene que estar libre para pasar a holding."""
        tope = max(usdt, self.caja_cuarto())
        return [i for i in self._atrapados() if tope + 1e-9 >= self.slots[i].costo]

    def chequear_cuarto(self, t: pd.Timestamp, precio: float):
        s4 = self.slots[3]
        if self.espera_cuenta and s4.abierto:
            if not self.esp[3]:
                return
            caja = sum(x["usdt"] for x in self.esp[3])
            atr = self._atrapados()
            for i in atr:
                if caja + 1e-9 >= self.slots[i].costo:
                    tramos, self.esp[3] = self.esp[3], []
                    self._reemplazar(i, tramos, t, precio, "4")
                    return
            if not atr:
                tramos, self.esp[3] = self.esp[3], []
                for x in tramos:
                    self._tercios(x)
            return
        if s4.abierto or not self.tr[3]:
            return
        caja = sum(x["usdt"] for x in self.tr[3])
        atr = self._atrapados()
        if atr:
            for i in atr:
                if caja + 1e-9 >= self.slots[i].costo:
                    tramos, self.tr[3] = self.tr[3], []
                    self._sync(3)
                    self._reemplazar(i, tramos, t, precio, "4")
                    return
            return
        tramos, self.tr[3] = self.tr[3], []
        self._sync(3)
        for x in tramos:
            self._tercios(x)

    def monto_holding(self) -> float:
        return self.pend_holding + self.reserva

    def comprar_holding(self, precio: float):
        monto = self.monto_holding()
        if monto + 1e-9 >= self.cfg.min_orden:
            qty, _ = self._comprar(monto, precio)
            self.holding_comprado(monto, qty)

    def holding_comprado(self, gastado: float, qty: float):
        """Una compra de holding que gastó `gastado` USDT ya ocurrió (simulada o real). En el real
        la orden va redondeada a centavos: lo que no se gastó sigue esperando en la reserva."""
        resto = self.monto_holding() - gastado
        self.holding_btc += qty
        self.st["holding_gastado"] += gastado
        self.pend_holding = 0.0
        self.reserva = resto if resto > 1e-9 else 0.0

    def _entrar(self, i: int, t: pd.Timestamp, precio: float):
        s = self.slots[i]
        self._sync(i)
        if s.abierto or s.nominal < self.cfg.min_orden:
            return
        qty, fill = self._comprar(s.nominal, precio)
        self.entrar_con_fill(i, t, s.nominal, qty, fill)

    def entrar_con_fill(self, i: int, t: pd.Timestamp, costo: float, qty: float, fill: float):
        super().entrar_con_fill(i, t, costo, qty, fill)
        self.st["entradas"] += 1

    def cerrar_con_fill(self, i: int, t: pd.Timestamp, neto: float, precio: float, motivo: str):
        s = self.slots[i]
        ganancia = neto - s.costo
        base = sum(x["usdt"] for x in self.tr[i])
        quedan = []
        for x in self.tr[i]:
            if ganancia < 0:
                x["usdt"] += ganancia * x["usdt"] / base
            if x["lote"] == PERM:
                quedan.append(x)
                continue
            x["ciclos"] += 1
            if x["ciclos"] >= self.max_ciclos:
                self.pend_holding += x["usdt"]
                self.st["usdt_sale_ciclos"] += x["usdt"]
                self.dias_4_ciclos.append((t - x["t0"]).total_seconds() / 86400)
            else:
                quedan.append(x)
        self.operaciones.append({"slot": i + 1, "entrada_t": s.desde, "salida_t": t, "entrada": s.entrada,
                                 "salida": precio, "qty": s.qty, "costo": s.costo, "neto": neto,
                                 "ganancia": ganancia, "motivo": motivo})
        self.st["realizado"] += ganancia
        self.st["ventas"] += 1
        if i == 3:
            self.st["ventas_4"] += 1
        s.abierto, s.qty, s.costo, s.armado = False, 0.0, 0.0, False
        self.atrapado[i] = False
        self.tr[i], espera = quedan, self.esp[i]
        self.esp[i] = []
        for x in espera:
            self._add(i, x)
        self._sync(i)
        if ganancia > 0:
            reinv = ganancia * REINV
            self.reserva += ganancia - reinv
            self._tercios({"lote": PERM, "usdt": reinv, "ciclos": 0, "t0": t, "p0": precio})

    def venta_parcial(self, i: int, t: pd.Timestamp, neto: float, qty: float, precio: float, motivo: str):
        """Sólo el real: la venta al TP se llenó en parte y el resto se canceló. La parte vendida
        realiza su ganancia (90/10 como una venta entera) y el slot sigue abierto con el resto, sin
        contar un ciclo: el ciclo se cuenta cuando se vende todo."""
        s = self.slots[i]
        frac = min(qty / s.qty, 1.0) if s.qty > 0 else 1.0
        costo_parte = s.costo * frac
        ganancia = neto - costo_parte
        self.operaciones.append({"slot": i + 1, "entrada_t": s.desde, "salida_t": t, "entrada": s.entrada,
                                 "salida": precio, "qty": qty, "costo": costo_parte, "neto": neto,
                                 "ganancia": ganancia, "motivo": motivo})
        self.st["realizado"] += ganancia
        # el costo de la parte vendida vuelve como USDT del slot (tramos − costo, ver `usdt_bot`)
        s.qty = max(s.qty - qty, 0.0)
        s.costo -= costo_parte
        if ganancia > 0:
            reinv = ganancia * REINV
            self.reserva += ganancia - reinv
            self._tercios({"lote": PERM, "usdt": reinv, "ciclos": 0, "t0": t, "p0": precio})

    def marcar_atrapados(self, t_cierre: pd.Timestamp, c: float):
        if not self.atrapar:
            return
        for i in range(3):
            s = self.slots[i]
            if s.abierto and not self.atrapado[i] and (
                    c <= s.entrada * (1 - self.caida) or t_cierre - s.desde >= pd.Timedelta(days=self.dias)):
                self.atrapado[i] = True
                self.st["n_atrapadas"] += 1

    def _senales(self, t_sig: pd.Timestamp, c: float, senal: bool):
        self.pendientes = []
        if not senal or self.ultimo_dia_entrada == (t_sig + pd.Timedelta(hours=1)).strftime("%Y-%m-%d"):
            return
        for i, s in enumerate(self.slots):
            if not s.abierto and s.nominal >= self.cfg.min_orden:
                self.pendientes = [i]
                return

    def vela(self, t: pd.Timestamp, o: float, h: float, l: float, c: float, senal: bool, atr_prev: float,
             rango_prev: float = float("nan"), depositos=(), compra: bool = True):
        """Procesa una vela cerrada completa con fills simulados. `depositos` = [(usdt, lote), …]
        que llegan en esta vela. `compra=False` es el filtro de tendencia del laboratorio: el
        depósito se registra (el DCA puro compra igual) y espera en `reserva_filtro`."""
        for s in self.slots:
            s.entro_en_vela = False
        self.comprar_holding(o)
        if compra and self.reserva_filtro:
            for monto, lote, t0, p0 in self.reserva_filtro:
                self.asignar(monto, lote, t, o, t0, p0)
            self.reserva_filtro = []
        for monto, lote in depositos:
            if compra:
                self.depositar(monto, lote, t, o)
            else:
                self.registrar(monto, o)
                self.reserva_filtro.append((monto, lote, t, o))
        self.chequear_cuarto(t, o)
        for i in self.pendientes:
            self._entrar(i, t, o)
        self.pendientes = []
        self.rango = float(rango_prev)
        self._salidas(t, o, h, l, c, float(atr_prev))
        self.marcar_atrapados(t + pd.Timedelta(hours=1), c)
        self._senales(t, c, senal)

    # --- contabilidad ---------------------------------------------------------------
    def caja(self) -> float:
        return (sum(x["usdt"] for i in range(N_SLOTS) if not self.slots[i].abierto for x in self.tr[i])
                + sum(x["usdt"] for e in self.esp for x in e) + self.pend_holding + self.reserva
                + sum(r[0] for r in self.reserva_filtro))

    def usdt_bot(self) -> float:
        """USDT que el bot tiene en la cuenta: la caja más lo que no se gastó al comprar un slot
        abierto (sus tramos menos el costo real; en la simulación es 0)."""
        sobra = sum(max(sum(x["usdt"] for x in self.tr[i]) - s.costo, 0.0)
                    for i, s in enumerate(self.slots) if s.abierto)
        return self.caja() + sobra

    def invertido(self) -> float:
        return sum(s.costo for s in self.slots if s.abierto)

    def latente(self, precio: float) -> float:
        """Pérdida (+) o ganancia (−) no realizada de lo abierto, a `precio`."""
        return sum(s.costo - s.qty * precio for s in self.slots if s.abierto)

    def valor_btc(self, precio: float) -> float:
        return self.holding_btc + sum(s.qty for s in self.slots if s.abierto) + self.caja() / precio

    def patrimonio_btc(self, precio: float) -> float:
        return self.holding_btc + self.btc_en_slots() + self.usdt_bot() / precio

    def conservacion(self) -> float:
        """depositado + realizado = caja + invertido + holding gastado + base de atrapadas."""
        st = self.st
        return (st["depositado"] + st["realizado"]
                - (self.caja() + self.invertido() + st["holding_gastado"] + st["base_atrapadas"]))

    def iniciar_cortes(self, capital: float, precio: float):
        """Punto de partida del primer corte: el capital y el precio del arranque (o de la migración)."""
        self.corte_previo = {"capital_bot": capital, "precio": precio}

    def corte_mensual(self, t: pd.Timestamp, precio: float) -> Dict:
        """Fila contable del ciclo que termina (día 3). No opera. El neto del ciclo es la ganancia
        realizada − el costo de las atrapadas reemplazadas − el cambio en la pérdida latente; su
        % es sobre el capital del bot al empezar el ciclo (`capital_inicio`)."""
        acum = {k: self.st[k] for k in ("realizado", "atrapadas_costo", "n_atrapadas", "depositado", "ventas",
                                        "entradas", "reemplazos_deposito", "reemplazos_4")}
        acum["latente"] = self.latente(precio)
        acum["capital_bot"] = self.usdt_bot() + self.invertido()
        prev = self.corte_previo
        fila = {k: acum[k] - prev.get(k, 0.0) for k in acum if k not in ("latente", "capital_bot")}
        fila.update({"ciclo": self.ciclo, "cierre": t, "precio_cierre": precio, "latente": acum["latente"],
                     "neto": fila["realizado"] - fila["atrapadas_costo"] - (acum["latente"] - prev.get("latente", 0.0)),
                     "capital_inicio": prev.get("capital_bot", float("nan")),
                     "precio_inicio": prev.get("precio", float("nan")),
                     "capital_bot": acum["capital_bot"], "holding_btc": self.holding_btc,
                     "bench_btc": self.bench_btc, "patrimonio_btc": self.patrimonio_btc(precio),
                     "atrapados_abiertos": sum(self.atrapado)})
        self.corte_previo = dict(acum, precio=precio)
        self.n_cierres += 1
        self.meses.append(fila)
        return fila

    # --- estado ---------------------------------------------------------------------
    def a_dict(self) -> Dict:
        from dataclasses import asdict
        return _a_json({
            "slots": [asdict(s) for s in self.slots], "tr": self.tr, "esp": self.esp, "atrapado": self.atrapado,
            "pend_holding": self.pend_holding, "reserva": self.reserva, "holding_btc": self.holding_btc,
            "bench_btc": self.bench_btc, "st": self.st, "corte_previo": self.corte_previo, "sig_lote": self.sig_lote,
            "ciclo": self.ciclo, "n_cierres": self.n_cierres, "ultimo_dia_entrada": self.ultimo_dia_entrada,
            "pendientes": self.pendientes, "rango": self.rango, "comisiones": self.comisiones,
            "max_ciclos": self.max_ciclos, "caida": self.caida, "dias": self.dias})

    @classmethod
    def desde_dict(cls, d: Dict, cfg: mm.Config) -> "MotorLotes":
        m = cls(cfg, max_ciclos=d["max_ciclos"], caida=d["caida"], dias=d["dias"])
        slots = []
        for s in d["slots"]:
            s = dict(s)
            s["desde"] = pd.Timestamp(s["desde"]) if s.get("desde") else None
            slots.append(mm.Slot(**s))
        m.slots = slots
        ts = lambda lst: [{**x, "t0": pd.Timestamp(x["t0"])} for x in lst]
        m.tr = [ts(x) for x in d["tr"]]
        m.esp = [ts(x) for x in d["esp"]]
        for k in ("atrapado", "pend_holding", "reserva", "holding_btc", "bench_btc", "corte_previo", "sig_lote",
                  "ciclo", "n_cierres", "ultimo_dia_entrada", "pendientes", "rango", "comisiones"):
            setattr(m, k, d[k])
        m.st.update(d["st"])
        return m


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


def migrar_desde_v04(motor_v04: Dict, cfg: mm.Config, t: pd.Timestamp, precio_inicio: float, capital: float,
                     max_ciclos: float = MAX_CICLOS) -> MotorLotes:
    """Pasa el estado del motor v0.4 (`papel.motor_a_dict`) a lotes, sin tocar el exchange: cada
    slot abierto es un tramo del lote inicial (0) por su nominal, con 0 ciclos contados desde `t`;
    un slot libre lleva su nominal como tramo; la ganancia reservada para holding pasa a la
    reserva. El DCA puro compra el capital inicial a `precio_inicio` (el del arranque del bot)."""
    m = MotorLotes(cfg, max_ciclos=max_ciclos)
    viejos = motor_v04["slots"]
    assert len(viejos) == 3, "el v0.4 tiene 3 slots"
    for i, s in enumerate(viejos):
        if not s.get("activo", True):
            raise ValueError(f"el slot {i + 1} está apagado: migrar antes de un cierre o reponerlo primero")
        slot = m.slots[i]
        slot.abierto, slot.costo, slot.qty, slot.entrada = s["abierto"], s["costo"], s["qty"], s["entrada"]
        slot.maximo, slot.armado = s["maximo"], s["armado"]
        slot.desde = pd.Timestamp(s["desde"]) if s.get("desde") else None
        m.tr[i] = [{"lote": 0, "usdt": float(s["nominal"]), "ciclos": 0, "t0": t, "p0": precio_inicio}]
        slot.nominal = float(s["nominal"])
    m.holding_btc = float(motor_v04["holding_btc"])
    m.reserva = float(motor_v04["usdt_para_holding"])
    m.ultimo_dia_entrada = motor_v04.get("ultimo_dia_entrada")
    m.rango = float(motor_v04.get("rango", float("nan")))
    m.comisiones = float(motor_v04.get("comisiones", 0.0))
    m.st["depositado"] = capital
    m.bench_btc = m._comprar_bench(capital, precio_inicio)[0]
    m.sig_lote = 1
    m.iniciar_cortes(capital, precio_inicio)
    return m
