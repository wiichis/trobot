"""Motor del bot BTC por rangos — el mismo para la simulación y para el live.

Especificación: `docs/reglas_bot_btc.md` v0.1. Señales en `pkg/btc_rangos_reglas.py`.
Reglas de ejecución fijadas el 22/10/2026, ANTES de correr ningún backtest:

Vela a vela (1 h, `date` = apertura UTC), en este orden:
1. Cierre mensual si la vela abre un ciclo nuevo (el ciclo empieza el día 3 a las 00:00
   UTC). Se ejecuta al precio de APERTURA de esa vela:
   a. Los slots abiertos (atrapados) pasan a holding sin operar. Costo de atrapada =
      (precio de entrada − precio de apertura) × cantidad.
   b. Llega el DCA del mes. Repone los slots atrapados o apagados en orden 1 → 2 → 3,
      cada uno hasta su tamaño nominal. Orden estricto: el primero que no alcanza a
      reponerse y los que siguen quedan apagados este ciclo.
   c. Con el sobrante del DCA, más la parte de la ganancia destinada a holding durante el
      ciclo, se compra BTC para holding (una orden de mercado).
   El benchmark DCA puro compra el DCA entero en el mismo momento y con los mismos costos.
2. Entradas pendientes (decididas al cierre de la vela anterior), a la APERTURA.
3. Salidas de cada slot abierto, intrabarra conservador:
   - S1 TP fijo: si abre sobre el TP vende a la apertura; si el high lo toca, al TP.
   - S2/S3 trailing: primero se mira el low contra el stop calculado con el máximo
     PREVIO (si abre debajo, vende a la apertura); recién después se actualiza el máximo
     con el high y se arma el trailing si el máximo llegó a +1,5 %. Un slot que entra en
     esta vela no puede salir por trailing en la misma vela.
   - Piso: nunca se vende por debajo del precio que devuelve el costo del slot, comisiones
     incluidas. Si el precio de salida quedaría debajo del piso, no vende y sigue abierto.
   Con `intrabar=False` todo se evalúa sólo con el close (corrida de sensibilidad).
4. Al cierre, señales para la vela siguiente:
   - Modo A: el slot libre de menor número entra si hay señal; como máximo 1 entrada por
     día UTC (contando el día de la ejecución).
   - Modo B: el slot 1 entra con señal. Con el slot 1 abierto, el 2 entra si el close
     está −X % bajo la entrada del slot 1, y el 3 a −2X %.

Costos: comisión 0,1 % por lado y slippage 0,05 % por lado (compra paga más, venta
recibe menos). Cantidades al step del exchange; un slot con menos USDT que la orden
mínima no entra.

Destino de la ganancia realizada (G): una parte se reinvierte, repartida en partes iguales
entre los 3 slots (sube su nominal), y el resto se reserva en USDT para comprar BTC de
holding en el próximo cierre mensual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

COMISION = 0.001
SLIPPAGE = 0.0005
STEP_BTC = 1e-6
MIN_ORDEN_USDT = 15.0  # minQty de BingX spot (0,0001826 BTC) a ~81.000 USDT, redondeado arriba

SALIDAS = {
    "S1_1.5": {"tipo": "S1", "tp": 0.015},
    "S1_2": {"tipo": "S1", "tp": 0.02},
    "S1_3": {"tipo": "S1", "tp": 0.03},
    "S2": {"tipo": "S2", "activa": 0.015, "retroceso": 0.006},
    "S3": {"tipo": "S3", "activa": 0.015, "atr_mult": 0.5},
}
GANANCIA = {"G1": 0.0, "G2": 1.0, "G3": 0.5, "G4": 0.9}  # fracción que se reinvierte en slots


@dataclass
class Config:
    slots: str = "A"            # "A" independientes, "B" escalonados
    escalon: float = 0.02       # X del modo B
    salida: str = "S2"
    ganancia: str = "G4"
    capital: float = 200.0
    n_slots: int = 3
    intrabar: bool = True
    comision: float = COMISION
    slippage: float = SLIPPAGE
    min_orden: float = MIN_ORDEN_USDT


@dataclass
class Slot:
    nominal: float
    abierto: bool = False
    activo: bool = True         # False = apagado hasta el próximo cierre mensual
    costo: float = 0.0          # USDT gastados en la compra (comisión incluida)
    qty: float = 0.0
    entrada: float = 0.0        # precio de fill de la compra
    maximo: float = 0.0
    armado: bool = False        # trailing activado
    desde: Optional[pd.Timestamp] = None
    entro_en_vela: bool = False


def ciclo_de(t: pd.Timestamp) -> str:
    """Ciclo mensual al que pertenece la vela: empieza el día 3 a las 00:00 UTC."""
    return (t - pd.Timedelta(days=2)).strftime("%Y-%m")


def _redondear(q: float) -> float:
    return np.floor(q / STEP_BTC + 1e-9) * STEP_BTC


@dataclass
class Motor:
    cfg: Config
    dca: Callable[[int], float]                 # monto del DCA del k-ésimo cierre (k = 1, 2, …)
    slots: List[Slot] = field(default_factory=list)
    holding_btc: float = 0.0
    usdt_para_holding: float = 0.0
    bench_btc: float = 0.0
    ciclo: Optional[str] = None
    n_cierres: int = 0
    ultimo_dia_entrada: Optional[str] = None
    pendientes: List[int] = field(default_factory=list)
    operaciones: List[Dict] = field(default_factory=list)
    meses: List[Dict] = field(default_factory=list)
    mes: Dict = field(default_factory=dict)
    comisiones: float = 0.0

    def __post_init__(self):
        if not self.slots:
            self.slots = [Slot(nominal=self.cfg.capital / self.cfg.n_slots) for _ in range(self.cfg.n_slots)]
        self._nuevo_mes()

    # --- contabilidad -------------------------------------------------------------
    def _nuevo_mes(self):
        self.mes = {"realizado": 0.0, "atrapadas_costo": 0.0, "n_atrapadas": 0, "ciclos": 0, "entradas": 0,
                    "comisiones": 0.0, "capital_inicio": self.capital_bot_nominal()}

    def capital_bot_nominal(self) -> float:
        return sum(s.nominal for s in self.slots)

    def usdt_bot(self) -> float:
        """USDT del bot: el nominal de los slots libres más lo que sobra en los abiertos o
        atrapados (nominal − costo)."""
        total = 0.0
        for s in self.slots:
            if s.abierto or not s.activo:
                total += max(s.nominal - s.costo, 0.0)
            else:
                total += s.nominal
        return total

    def btc_en_slots(self) -> float:
        return sum(s.qty for s in self.slots if s.abierto)

    # --- órdenes ------------------------------------------------------------------
    def _comprar(self, usdt: float, precio: float) -> tuple[float, float]:
        fill = precio * (1 + self.cfg.slippage)
        qty = _redondear(usdt * (1 - self.cfg.comision) / fill)
        com = usdt * self.cfg.comision
        self.comisiones += com
        self.mes["comisiones"] += com
        return qty, fill

    def _vender(self, qty: float, precio: float) -> float:
        bruto = qty * precio * (1 - self.cfg.slippage)
        com = bruto * self.cfg.comision
        self.comisiones += com
        self.mes["comisiones"] += com
        return bruto - com

    def piso(self, s: Slot) -> float:
        """Precio de mercado mínimo al que vender devuelve el costo del slot."""
        return s.costo / (s.qty * (1 - self.cfg.slippage) * (1 - self.cfg.comision))

    # --- pasos de la vela -----------------------------------------------------------
    def cierre_mensual(self, t: pd.Timestamp, precio: float):
        self.n_cierres += 1
        for s in self.slots:
            if s.abierto:
                self.mes["atrapadas_costo"] += (s.entrada - precio) * s.qty
                self.mes["n_atrapadas"] += 1
                self.holding_btc += s.qty
                s.abierto, s.activo, s.qty, s.armado = False, False, 0.0, False
                # s.costo queda: es lo que hay que reponer
        dca = float(self.dca(self.n_cierres))
        disponible = dca
        repuesto = 0.0
        cortado = False
        for s in self.slots:
            if s.activo:
                continue
            falta = s.costo
            if not cortado and disponible + 1e-9 >= falta:
                disponible -= falta
                repuesto += falta
                s.activo, s.costo = True, 0.0
            else:
                cortado = True
        compra_holding = disponible + self.usdt_para_holding
        if compra_holding > 0:
            qty, _ = self._comprar(compra_holding, precio)
            self.holding_btc += qty
        self.usdt_para_holding = 0.0
        qty_b, _ = self._comprar_bench(dca, precio)
        self.bench_btc += qty_b
        self.mes.update({"ciclo": self.ciclo, "cierre": t, "precio_cierre": precio, "dca": dca,
                         "repuesto": repuesto, "compra_holding": compra_holding,
                         "slots_apagados": sum(not s.activo for s in self.slots),
                         "holding_btc": self.holding_btc, "bench_btc": self.bench_btc})
        self.meses.append(self.mes)
        self._nuevo_mes()

    def _comprar_bench(self, usdt: float, precio: float) -> tuple[float, float]:
        fill = precio * (1 + self.cfg.slippage)
        return _redondear(usdt * (1 - self.cfg.comision) / fill), fill

    def _entrar(self, i: int, t: pd.Timestamp, precio: float):
        s = self.slots[i]
        if s.abierto or not s.activo or s.nominal < self.cfg.min_orden:
            return
        qty, fill = self._comprar(s.nominal, precio)
        s.abierto, s.costo, s.qty, s.entrada, s.maximo = True, s.nominal, qty, fill, fill
        s.armado, s.desde, s.entro_en_vela = False, t, True
        self.ultimo_dia_entrada = t.strftime("%Y-%m-%d")
        self.mes["entradas"] += 1

    def _cerrar(self, i: int, t: pd.Timestamp, precio: float, motivo: str):
        s = self.slots[i]
        neto = self._vender(s.qty, precio)
        ganancia = neto - s.costo
        reinv = ganancia * GANANCIA[self.cfg.ganancia] if ganancia > 0 else 0.0
        self.usdt_para_holding += ganancia - reinv
        self.operaciones.append({"slot": i + 1, "entrada_t": s.desde, "salida_t": t, "entrada": s.entrada,
                                 "salida": precio, "qty": s.qty, "costo": s.costo, "neto": neto,
                                 "ganancia": ganancia, "motivo": motivo})
        self.mes["realizado"] += ganancia
        self.mes["ciclos"] += 1
        s.abierto, s.qty, s.costo, s.armado = False, 0.0, 0.0, False
        for x in self.slots:
            x.nominal += reinv / len(self.slots)

    def _salidas(self, t, o, h, l, c, atr_prev: float):
        p = SALIDAS[self.cfg.salida]
        for i, s in enumerate(self.slots):
            if not s.abierto:
                continue
            piso = self.piso(s)
            if not self.cfg.intrabar:
                o_, h_, l_ = c, c, c
            else:
                o_, h_, l_ = o, h, l
            if p["tipo"] == "S1":
                tp = max(s.entrada * (1 + p["tp"]), piso)
                if o_ >= tp:
                    self._cerrar(i, t, o_, "tp")
                elif h_ >= tp:
                    self._cerrar(i, t, tp, "tp")
                continue
            # trailing
            if s.armado and not s.entro_en_vela:
                if p["tipo"] == "S2":
                    stop = s.maximo * (1 - p["retroceso"])
                else:
                    stop = s.maximo - p["atr_mult"] * atr_prev
                if np.isnan(stop):  # ATR todavía sin calentar: no hay nivel de trailing
                    stop = -np.inf
                else:
                    stop = max(stop, piso)
                if o_ <= stop:
                    if o_ >= piso:
                        self._cerrar(i, t, o_, "trailing")
                        continue
                elif l_ <= stop:
                    self._cerrar(i, t, stop, "trailing")
                    continue
            s.maximo = max(s.maximo, h_)
            if not s.armado and s.maximo >= s.entrada * (1 + p["activa"]):
                s.armado = True

    def _senales(self, t_sig: pd.Timestamp, c: float, senal: bool):
        """Decide al cierre qué slots entran en la apertura siguiente."""
        self.pendientes = []
        t_ejec = t_sig + pd.Timedelta(hours=1)
        if self.cfg.slots == "A":
            if not senal or self.ultimo_dia_entrada == t_ejec.strftime("%Y-%m-%d"):
                return
            for i, s in enumerate(self.slots):
                if not s.abierto and s.activo:
                    self.pendientes = [i]
                    return
            return
        s1 = self.slots[0]
        if not s1.abierto:
            if senal and s1.activo:
                self.pendientes = [0]
            return
        for k in range(1, len(self.slots)):
            s = self.slots[k]
            if not s.abierto and s.activo and c <= s1.entrada * (1 - k * self.cfg.escalon):
                self.pendientes.append(k)

    def vela(self, t: pd.Timestamp, o: float, h: float, l: float, c: float, senal: bool, atr_prev: float):
        """Procesa una vela cerrada completa. `senal` es la señal calculada con ESTA vela
        (se ejecuta en la siguiente); `atr_prev` es el ATR de la vela anterior."""
        ciclo = ciclo_de(t)
        if self.ciclo is None:
            self.ciclo = ciclo
        elif ciclo != self.ciclo:
            self.cierre_mensual(t, o)
            self.ciclo = ciclo
        for s in self.slots:
            s.entro_en_vela = False
        for i in self.pendientes:
            self._entrar(i, t, o)
        self.pendientes = []
        self._salidas(t, o, h, l, c, atr_prev)
        self._senales(t, c, senal)

    def finalizar(self, t: pd.Timestamp, precio: float):
        """Cierra la contabilidad del último ciclo al final de la simulación: marca el costo
        de las atrapadas a `precio` sin mover el estado (no hay DCA ni compra de holding)."""
        for s in self.slots:
            if s.abierto:
                self.mes["atrapadas_costo"] += (s.entrada - precio) * s.qty
                self.mes["n_atrapadas"] += 1
        self.mes.update({"ciclo": self.ciclo, "cierre": t, "precio_cierre": precio, "dca": 0.0, "repuesto": 0.0,
                         "compra_holding": 0.0, "slots_apagados": sum(not s.activo for s in self.slots),
                         "holding_btc": self.holding_btc, "bench_btc": self.bench_btc, "final": True})
        self.meses.append(self.mes)

    def patrimonio_btc(self, precio: float) -> float:
        return self.holding_btc + self.btc_en_slots() + (self.usdt_bot() + self.usdt_para_holding) / precio


@dataclass
class Resultado:
    motor: Motor
    precio_inicio: float
    precio_fin: float
    bench_b_extra_btc: float   # los 200 USDT iniciales comprados en BTC el primer día

    @property
    def patrimonio_btc(self) -> float:
        return self.motor.patrimonio_btc(self.precio_fin)

    @property
    def bench_a_btc(self) -> float:
        return self.motor.bench_btc + self.motor.cfg.capital / self.precio_fin

    @property
    def bench_b_btc(self) -> float:
        return self.motor.bench_btc + self.bench_b_extra_btc


def simular(velas: pd.DataFrame, senal: pd.Series, atr: pd.Series, cfg: Config,
            dca: Callable[[int], float], desde: pd.Timestamp, hasta: pd.Timestamp) -> Resultado:
    """Corre el motor sobre [desde, hasta). `senal` y `atr` deben estar calculados sobre la
    serie completa (los indicadores sólo miran hacia atrás) e indexados como `velas`."""
    sl = velas.loc[(velas.index >= desde) & (velas.index < hasta)]
    sen = senal.reindex(sl.index).fillna(False).to_numpy()
    atr_prev = atr.shift(1).reindex(sl.index).to_numpy()
    m = Motor(cfg=cfg, dca=dca)
    o_, h_, l_, c_ = (sl[k].to_numpy() for k in ("open", "high", "low", "close"))
    for j, t in enumerate(sl.index):
        m.vela(t, o_[j], h_[j], l_[j], c_[j], bool(sen[j]), float(atr_prev[j]))
    m.finalizar(sl.index[-1] + pd.Timedelta(hours=1), float(c_[-1]))
    p0 = float(o_[0])
    bench_b = _redondear(cfg.capital * (1 - cfg.comision) / (p0 * (1 + cfg.slippage)))
    return Resultado(motor=m, precio_inicio=p0, precio_fin=float(c_[-1]), bench_b_extra_btc=bench_b)
