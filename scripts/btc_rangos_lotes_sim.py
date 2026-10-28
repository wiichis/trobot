"""Bot BTC por rangos con lotes de 4 ciclos y reemplazo de atrapados (27/10/2026).

Pedido del usuario: cada depósito de USDT es un lote que se reparte entre los 3 slots. Cada
tramo hace como máximo 4 ciclos (compra → venta con ganancia) y después su capital original
compra BTC para holding (subcuenta → billetera fría). La ganancia es del bot: 90 % se
reinvierte en los slots, 10 % compra BTC para holding (G4). No hay corte el día 3: un slot
queda atrapado por caída o por tiempo, y se reemplaza cuando entra USDT que cubre su costo; si
el USDT no alcanza, opera como un 4.º slot temporal hasta que cubra a un atrapado.

FIJADO ANTES DE CORRER (respuestas del usuario del 27/10):
- 4 ciclos por tramo; 3 slots compartidos (cada slot guarda una lista de tramos); 1 entrada
  por día en total; entrada E4_0.5 y salida S4_0.75 con piso de costo, como el bot real.
- Atrapado: al cierre, un slot abierto con close ≤ entrada × 0,90 o ≥ 30 días abierto. Es
  pegajoso (sale sólo si vende o si lo reemplazan) y mantiene su TP.
- Depósito con atrapados: reemplaza al atrapado más barato que cubra por sí solo; el BTC del
  atrapado pasa a holding (costo de atrapada = costo − qty × precio). El sobrante sigue la
  misma regla; lo que no cubre va al 4.º slot. Sin atrapados, el depósito va en tercios a los
  slots 1-3 (si un slot está abierto, su tercio espera y entra con la próxima entrada).
- 4.º slot: cerrado y con USDT ≥ costo de un atrapado → lo reemplaza (sus tramos se mudan con
  sus ciclos). Sin atrapados, sus tramos se reparten en tercios entre los slots 1-3.
- Variante B (`espera_cuenta`, agregada después de ver la A): el hueco de la regla es qué pasa
  con los depósitos si el 4.º slot está abierto. En la A esperan a que venda (en un calendario,
  10.561 USDT ociosos 2 años con el 4.º abierto desde 2021). En la B el USDT que espera en el
  4.º cuenta para reemplazar a un atrapado aunque el 4.º esté abierto, y sin atrapados se
  reparte en tercios de inmediato.
- Los tramos que cumplen 4 ciclos y el 10 % de la ganancia compran BTC de holding a la
  apertura de la vela siguiente, a mercado, cuando suman al menos la orden mínima (15 USDT).
- Depósitos: cada ciclo mensual (día 3 → día 3), total U(100, 900) en 1-3 compras de ≥ 50
  USDT, cada una en una vela al azar del ciclo. 200 calendarios.
- Comparaciones, con los mismos depósitos: (1) DCA puro: cada depósito compra BTC en su vela;
  (2) regla actual: `Motor` A · E4_0.5 · S4_0.75 · G4 con reposición parcial, capital inicial
  = depósitos del primer ciclo y el DCA del día 3 = los del ciclo que empieza; (3) placebo:
  las reglas nuevas con entradas al azar a la frecuencia de la señal, 50 calendarios.
- Períodos: continuo 2021-01-03 → 2026-10-03, y de cero 2021-01-03 → 2024-01-03 y
  2024-01-03 → 2026-10-03.
- Costos: comisión 0,1 % y slippage 0,05 % por lado; step 1e-6 BTC.

MEJORAS (28/10/2026), fijadas antes de correr (plan aprobado por el usuario):
- Ciclos por lote ∈ {1, 2, 4} (`b1`, `b2`, `b4`; `b4` = variante B). Motivo: medido contra el
  precio del día de su depósito, la entrada atrapada sale igual que el DCA en la 1.ª compra del
  dinero (−0,1 %) y empeora con cada ciclo previo (+2,6 / +4,9 / +8,8 % tras 1 / 2 / 3).
- Filtro de tendencia a los depósitos (`fb1`, `fb2`, `fb4`): ventana COMPRA si el último cierre
  diario está sobre la media de 100 días (regla T1 de `pkg/btc_alertas.py`, `rr.regimen_compra`).
  En VENTA el depósito espera en una reserva que no opera (el DCA compra igual ese día) y en la
  primera vela de COMPRA entra al bot como lote. Las entradas del bot no se bloquean.
- `dca_filtro`: el filtro sin bot (en COMPRA compra al instante; en VENTA espera). También con
  velas diarias 2017-12-03 → 2026-10-03 (`d_*`), para sumar la caída de 2018.
- Placebos (50): `b*` con entradas al azar; `fb*` y `dca_filtro` con la serie de ventanas
  desplazada en el tiempo un número de días al azar (conserva cuánto tiempo pasa en VENTA y
  cuántos cambios hay, no cuándo).
- Pasa si cumple todo: (1) mediana de la ventaja de BTC > 0 en 2021-2023 y en 2024-2026 (y en
  2017-2020 para `dca_filtro`); (2) gana en ≥ 60 % de los calendarios en cada tramo; (3) su
  mediana supera al percentil 90 de su placebo en el continuo; (4) ningún mes explica más del
  50 % de los aumentos de ventaja (mediana entre calendarios).

Uso: python3 scripts/btc_rangos_lotes_sim.py [--procesos 8] [--semillas 200] [--prueba]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "archivos" / "btc_rangos" / "lotes_sim"

_spec = importlib.util.spec_from_file_location("v04", REPO / "scripts" / "btc_rangos_v04.py")
v04 = importlib.util.module_from_spec(_spec)
sys.modules["v04"] = v04
_spec.loader.exec_module(v04)
rr, mm = v04.rr, v04.mm

ENTRADA, SALIDA = "E4_0.5", "S4_0.75"
MAX_CICLOS = 4
CAIDA_ATRAPADO = 0.10
DIAS_ATRAPADO = 30
REINV = mm.GANANCIA["G4"]
PERIODOS = {
    "continuo": (pd.Timestamp("2021-01-03"), pd.Timestamp("2026-10-03")),
    "2021-2023": (pd.Timestamp("2021-01-03"), pd.Timestamp("2024-01-03")),
    "2024-2026": (pd.Timestamp("2024-01-03"), pd.Timestamp("2026-10-03")),
}
PERIODOS_D = {  # velas diarias, sólo para `dca_filtro`; la media de 100 días está lista desde nov-17
    "d_continuo": (pd.Timestamp("2017-12-03"), pd.Timestamp("2026-10-03")),
    "d_2017-2020": (pd.Timestamp("2017-12-03"), pd.Timestamp("2021-01-03")),
    "d_2021-2023": (pd.Timestamp("2021-01-03"), pd.Timestamp("2024-01-03")),
    "d_2024-2026": (pd.Timestamp("2024-01-03"), pd.Timestamp("2026-10-03")),
}
COSTO_COMPRA = (1 + mm.SLIPPAGE) / (1 - mm.COMISION)  # precio efectivo por BTC = precio × COSTO_COMPRA
N_PLACEBO = 50
PERM = -1  # lote de los tramos de ganancia reinvertida: no tienen límite de ciclos


class MotorLotes(mm.Motor):
    """Slots 0-2 fijos y el 3 temporal. Cada slot guarda tramos {lote, usdt, ciclos, t0}."""

    def __init__(self, cfg: mm.Config, max_ciclos: float = MAX_CICLOS, atrapar: bool = True,
                 espera_cuenta: bool = False):
        super().__init__(cfg=cfg, dca=lambda k: 0.0)
        self.max_ciclos, self.atrapar, self.espera_cuenta = max_ciclos, atrapar, espera_cuenta
        self.tr = [[] for _ in range(4)]
        self.esp = [[] for _ in range(4)]
        self.atrapado = [False] * 4
        self.pend_holding = 0.0
        self.reserva = 0.0
        self.reserva_filtro: list = []  # depósitos que esperan la ventana COMPRA: (monto, lote, t, precio)
        self.bench_btc = 0.0
        self.st = {"atr_usdt": 0.0, "atr_btc_bot": 0.0, "atr_btc_dca": 0.0,"depositado": 0.0, "realizado": 0.0, "atrapadas_costo": 0.0, "n_atrapadas": 0,
                   "reemplazos_deposito": 0, "reemplazos_4": 0, "holding_gastado": 0.0, "base_atrapadas": 0.0,
                   "usdt_sale_ciclos": 0.0, "usdt_sale_atrapado": 0.0, "ocioso": 0.0, "total": 0.0,
                   "ventas_4": 0}
        self.dias_4_ciclos: list = []

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
        s.abierto, s.qty, s.costo, s.armado = False, 0.0, 0.0, False
        self.atrapado[i] = False
        self.tr[i], viejos = [], self.esp[i]
        self.esp[i] = []
        for x in nuevos + viejos:
            self._add(i, x)

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

    def comprar_holding(self, precio: float):
        monto = self.pend_holding + self.reserva
        if monto + 1e-9 >= self.cfg.min_orden:
            qty, _ = self._comprar(monto, precio)
            self.holding_btc += qty
            self.st["holding_gastado"] += monto
            self.pend_holding = self.reserva = 0.0

    def _entrar(self, i: int, t: pd.Timestamp, precio: float):
        s = self.slots[i]
        self._sync(i)
        if s.abierto or s.nominal < self.cfg.min_orden:
            return
        qty, fill = self._comprar(s.nominal, precio)
        self.entrar_con_fill(i, t, s.nominal, qty, fill)

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

    def marcar_atrapados(self, t_cierre: pd.Timestamp, c: float):
        if not self.atrapar:
            return
        for i in range(3):
            s = self.slots[i]
            if s.abierto and not self.atrapado[i] and (
                    c <= s.entrada * (1 - CAIDA_ATRAPADO) or t_cierre - s.desde >= pd.Timedelta(days=DIAS_ATRAPADO)):
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

    # --- contabilidad ---------------------------------------------------------------
    def caja(self) -> float:
        return (sum(x["usdt"] for i in range(4) if not self.slots[i].abierto for x in self.tr[i])
                + sum(x["usdt"] for e in self.esp for x in e) + self.pend_holding + self.reserva
                + sum(r[0] for r in self.reserva_filtro))

    def invertido(self) -> float:
        return sum(s.costo for s in self.slots if s.abierto)

    def valor_btc(self, precio: float) -> float:
        return self.holding_btc + sum(s.qty for s in self.slots if s.abierto) + self.caja() / precio

    def conservacion(self) -> float:
        """depositado + realizado = caja + invertido + holding gastado + base de atrapadas."""
        st = self.st
        return (st["depositado"] + st["realizado"]
                - (self.caja() + self.invertido() + st["holding_gastado"] + st["base_atrapadas"]))


def calendario(index: pd.DatetimeIndex, desde: pd.Timestamp, hasta: pd.Timestamp, rng) -> dict:
    """{posición de la vela: [(monto, lote), …]} y la suma por ciclo, en orden."""
    deps: dict = {}
    sumas = []
    lote = 0
    inicio = desde
    while inicio < hasta:
        fin = min(inicio + pd.DateOffset(months=1), hasta)
        pos = np.flatnonzero((index >= inicio) & (index < fin))
        total = round(float(rng.uniform(100, 900)), 2)
        k = min(int(rng.integers(1, 4)), int(total // 50))
        partes = np.round(50 + (total - 50 * k) * rng.dirichlet(np.ones(k)), 2)
        partes[-1] = round(total - partes[:-1].sum(), 2)
        for p, j in zip(partes, np.sort(rng.choice(pos, size=k, replace=False))):
            deps.setdefault(int(j), []).append((float(p), lote))
            lote += 1
        sumas.append(total)
        inicio = fin
    return {"deps": deps, "sumas": sumas}


_V = _IND = _SEN = _REG = _D = _REG_D = None


def _init():
    global _V, _IND, _SEN, _REG, _D, _REG_D
    _V = pd.read_csv(REPO / "archivos/btc_rangos/velas_1h.csv", parse_dates=["date"], index_col="date")
    _IND = rr.indicadores(_V)
    _D = v04.diario_ohlc(_V)
    _IND["rango"] = rr.rango_diario(_D, _V.index)
    _SEN = rr.senal_entrada(_V, _IND, ENTRADA, False)
    # `regimen_compra` es lo que se sabe al CIERRE de cada vela; al abrir la vela vale el de la anterior
    _REG = rr.regimen_compra(v04.bt.cierre_diario(_V), _V.index).shift(1, fill_value=False)
    c = _D["close"]
    _REG_D = (c > c.rolling(rr.N_REGIMEN).mean()).shift(1, fill_value=False)  # se sabe al abrir el día


def _fin_de_ciclo(index: pd.DatetimeIndex) -> np.ndarray:
    desplazado = index - pd.Timedelta(days=2)  # el ciclo empieza el día 3, como `mm.ciclo_de`
    ciclos = np.asarray(desplazado.year * 12 + desplazado.month)
    return np.append(ciclos[1:] != ciclos[:-1], True)


def correr_lotes(sl: pd.DataFrame, sen: np.ndarray, atr_prev: np.ndarray, rango_prev: np.ndarray, deps: dict,
                 max_ciclos: float = MAX_CICLOS, atrapar: bool = True, espera_cuenta: bool = False,
                 regimen: np.ndarray | None = None) -> tuple[MotorLotes, list]:
    cfg = mm.Config(slots="A", salida=SALIDA, ganancia="G4", capital=0.0, n_slots=4, cierre_forzado=False)
    m = MotorLotes(cfg, max_ciclos=max_ciclos, atrapar=atrapar, espera_cuenta=espera_cuenta)
    o_, h_, l_, c_ = (sl[k].to_numpy() for k in ("open", "high", "low", "close"))
    fin = _fin_de_ciclo(sl.index)
    fotos = []
    hora = pd.Timedelta(hours=1)
    for j, t in enumerate(sl.index):
        o = float(o_[j])
        for s in m.slots:
            s.entro_en_vela = False
        m.comprar_holding(o)
        compra = regimen is None or bool(regimen[j])
        if compra and m.reserva_filtro:
            for monto, lote, t0, p0 in m.reserva_filtro:
                m.asignar(monto, lote, t, o, t0, p0)
            m.reserva_filtro = []
        for monto, lote in deps.get(j, ()):
            if compra:
                m.depositar(monto, lote, t, o)
            else:
                m.registrar(monto, o)
                m.reserva_filtro.append((monto, lote, t, o))
        m.chequear_cuarto(t, o)
        for i in m.pendientes:
            m._entrar(i, t, o)
        m.pendientes = []
        m.rango = float(rango_prev[j])
        m._salidas(t, o, h_[j], l_[j], c_[j], float(atr_prev[j]))
        c = float(c_[j])
        m.marcar_atrapados(t + hora, c)
        caja = m.caja()
        m.st["ocioso"] += caja
        m.st["total"] += caja + m.invertido()
        m._senales(t, c, bool(sen[j]))
        if fin[j]:
            fotos.append((t + hora, m.valor_btc(c), m.bench_btc, c))
    return m, fotos


def dca_filtro(v: pd.DataFrame, regimen: np.ndarray, deps: dict, paso: pd.Timedelta) -> dict:
    """Cada depósito compra BTC a la apertura de la primera vela en COMPRA desde que llega (la suya
    si ya está en COMPRA). El DCA puro lo compra en su vela. Mismos costos y redondeo."""
    o_, c_ = v["open"].to_numpy(), v["close"].to_numpy()
    n = len(v)
    siguiente = np.full(n + 1, n)
    for j in range(n - 1, -1, -1):
        siguiente[j] = j if regimen[j] else siguiente[j + 1]
    filas = []  # (llega, compra o n, usdt, btc, btc_dca)
    for j, lst in deps.items():
        k = siguiente[j]
        for monto, _ in lst:
            btc = mm._redondear(monto * (1 - mm.COMISION) / (float(o_[k]) * (1 + mm.SLIPPAGE))) if k < n else 0.0
            dca = mm._redondear(monto * (1 - mm.COMISION) / (float(o_[j]) * (1 + mm.SLIPPAGE)))
            filas.append((j, k, monto, btc, dca))
    f = np.array(filas)
    fotos = []
    for e in np.flatnonzero(_fin_de_ciclo(v.index)):
        llego, compro = f[:, 0] <= e, f[:, 1] <= e
        espera = llego & ~compro
        fotos.append((v.index[e] + paso, f[compro, 3].sum() + f[espera, 2].sum() / c_[e], f[llego, 4].sum(), c_[e]))
    espera_fin = f[:, 1] >= n
    ocioso = (np.minimum(f[:, 1], n) - f[:, 0]).sum() / (n - f[:, 0]).sum() * 100  # USDT × vela esperando
    return {"bot_btc": f[:, 3].sum() + f[espera_fin, 2].sum() / c_[-1], "bench_btc": f[:, 4].sum(),
            "fotos": fotos, "ocioso_pct": ocioso, "usdt_espera_fin": f[espera_fin, 2].sum()}


def _desplazar(reg: pd.Series, semilla: int, por_dia: int) -> pd.Series:
    """Placebo del filtro: la misma serie de ventanas desplazada en el tiempo un número de días al azar."""
    dias = len(reg) // por_dia
    k = int(np.random.default_rng(20_000 + semilla).integers(60, dias - 60)) * por_dia
    return pd.Series(np.roll(reg.to_numpy(), k), index=reg.index)


def correr(args) -> dict:
    periodo, semilla, modo = args
    diario = periodo.startswith("d_")
    desde, hasta = (PERIODOS_D if diario else PERIODOS)[periodo]
    v = _D if diario else _V
    sl = v.loc[(v.index >= desde) & (v.index < hasta)]
    semilla_cal = [semilla, 10 + list(PERIODOS_D).index(periodo)] if diario else [semilla, list(PERIODOS).index(periodo)]
    cal = calendario(sl.index, desde, hasta, np.random.default_rng(semilla_cal))
    out = {"periodo": periodo, "semilla": semilla, "modo": modo, "depositado": sum(cal["sumas"])}
    reg = _REG_D if diario else _REG
    if modo.startswith("placebo_r_"):
        reg = _desplazar(reg, semilla, 1 if diario else 24)
    reg_a = reg.reindex(sl.index).fillna(False).to_numpy(dtype=bool)
    if modo.endswith("dca_filtro"):
        r = dca_filtro(sl, reg_a, cal["deps"], pd.Timedelta(days=1) if diario else pd.Timedelta(hours=1))
        out.update({k: r[k] for k in ("bot_btc", "bench_btc", "ocioso_pct", "usdt_espera_fin")})
        out["ventaja_pct"] = (out["bot_btc"] / out["bench_btc"] - 1) * 100
        if periodo in ("continuo", "d_continuo") and not modo.startswith("placebo"):
            out["fotos"] = r["fotos"]
        return out
    sen = _SEN
    if modo.startswith("placebo_b"):
        p = _SEN.loc[sl.index].mean()
        sen = pd.Series(np.random.default_rng(10_000 + semilla).random(len(_V)) < p, index=_V.index)
    sen_a = sen.reindex(sl.index).fillna(False).to_numpy()
    atr_prev = _IND["atr"].shift(1).reindex(sl.index).to_numpy()
    rango_prev = _IND["rango"].shift(1).reindex(sl.index).to_numpy()
    p_fin = float(sl["close"].iloc[-1])
    if modo == "actual":
        sumas = cal["sumas"]
        cfg = mm.Config(slots="A", salida=SALIDA, ganancia="G4", capital=sumas[0], reposicion_parcial=True)
        r = mm.simular(_V, sen, _IND["atr"], cfg, lambda k: sumas[k] if k < len(sumas) else 0.0, desde, hasta,
                       rango=_IND["rango"])
        me = pd.DataFrame(r.motor.meses)
        # el benchmark es el mismo DCA puro de los depósitos al azar, no el del día 3
        o_ = sl["open"].to_numpy()
        bench = sum(mm._redondear(monto * (1 - mm.COMISION) / (float(o_[j]) * (1 + mm.SLIPPAGE)))
                    for j, lst in cal["deps"].items() for monto, _ in lst)
        out.update({"bot_btc": r.patrimonio_btc, "bench_btc": bench, "holding_btc": r.motor.holding_btc,
                    "realizado": float(me["realizado"].sum()), "atrapadas_costo": float(me["atrapadas_costo"].sum()),
                    "latente_fin": 0.0, "n_atrapadas": int(me["n_atrapadas"].sum()),
                    "operaciones": len(r.motor.operaciones)})
        out["ventaja_pct"] = (out["bot_btc"] / out["bench_btc"] - 1) * 100
        return out
    filtro = "fb" in modo
    m, fotos = correr_lotes(sl, sen_a, atr_prev, rango_prev, cal["deps"], max_ciclos=int(modo[-1]),
                            espera_cuenta=True, regimen=reg_a if filtro else None)
    err = m.conservacion()
    if abs(err) > 1e-6 * max(m.st["depositado"], 1.0):
        raise AssertionError(f"conservación rota en {periodo}/{semilla}/{modo}: {err}")
    st = m.st
    latente = sum(s.costo - s.qty * p_fin for s in m.slots if s.abierto)
    out.update({"bot_btc": m.valor_btc(p_fin), "bench_btc": m.bench_btc, "holding_btc": m.holding_btc,
                "realizado": st["realizado"], "atrapadas_costo": st["atrapadas_costo"], "latente_fin": latente,
                "n_atrapadas": st["n_atrapadas"], "reemplazos_deposito": st["reemplazos_deposito"],
                "reemplazos_4": st["reemplazos_4"], "ventas_4": st["ventas_4"], "operaciones": len(m.operaciones),
                "usdt_sale_ciclos": st["usdt_sale_ciclos"], "usdt_sale_atrapado": st["usdt_sale_atrapado"],
                "dias_4_ciclos_mediana": float(np.median(m.dias_4_ciclos)) if m.dias_4_ciclos else np.nan,
                "ocioso_pct": st["ocioso"] / st["total"] * 100 if st["total"] else np.nan,
                "atr_usdt": st["atr_usdt"],
                "atr_vs_dep_pct": (st["atr_btc_bot"] / st["atr_btc_dca"] - 1) * 100 if st["atr_btc_dca"] else np.nan,
                "atrapados_fin": sum(m.atrapado), "caja_fin": m.caja()})
    out["ventaja_pct"] = (out["bot_btc"] / out["bench_btc"] - 1) * 100
    if periodo == "continuo" and not modo.startswith("placebo"):
        out["fotos"] = fotos
    return out


# --- pruebas ------------------------------------------------------------------------
def _velas_sinteticas(n_dias: int, tocar_tp: bool = True) -> pd.DataFrame:
    """Precio plano en 100; cada día, la vela de las 12:00 sube el high a 102 (toca el TP)."""
    idx = pd.date_range("2025-01-03", periods=n_dias * 24, freq="h")
    v = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    if tocar_tp:
        v.loc[v.index.hour == 12, "high"] = 102.0
    return v


def prueba() -> None:
    com, slip = mm.COMISION, mm.SLIPPAGE
    # 1) caso a mano: 1 depósito de 300, señal a las 00:00 de cada día, R = 2 % → TP = +1,5 %
    v = _velas_sinteticas(6)
    sen = (v.index.hour == 0)
    rango = np.full(len(v), 0.02)
    m, _ = correr_lotes(v, sen, np.full(len(v), np.nan), rango, {0: [(300.0, 0)]})
    assert [op["slot"] for op in m.operaciones] == [1, 1, 1, 1, 2, 2], m.operaciones
    fill = 100 * (1 + slip)
    q1 = mm._redondear(100 * (1 - com) / fill)
    tp = fill * 1.015
    g1 = q1 * tp * (1 - slip) * (1 - com) - 100
    assert abs(m.operaciones[0]["ganancia"] - g1) < 1e-9, (m.operaciones[0]["ganancia"], g1)
    # tras el 4.º ciclo el tramo del lote (100 USDT) se compra en holding a la apertura siguiente
    q_hold = mm._redondear((100 + 0.1 * sum(op["ganancia"] for op in m.operaciones[:4])) * (1 - com) / fill)
    assert m.st["usdt_sale_ciclos"] == 100.0, m.st
    assert abs(m.holding_btc - q_hold) < 1e-12, (m.holding_btc, q_hold)
    assert abs(m.conservacion()) < 1e-9
    print(f"✔ caso a mano: 4 ciclos → 100 USDT del lote a holding ({m.holding_btc:.6f} BTC)")

    # 2) atrapado y reemplazo: sin TP, cae −12 %; el depósito siguiente cubre y lo reemplaza
    v = _velas_sinteticas(3, tocar_tp=False)
    v.iloc[30:, :] = 88.0
    sen = np.zeros(len(v), bool)
    sen[0] = True
    m, _ = correr_lotes(v, sen, np.full(len(v), np.nan), np.full(len(v), 0.02), {0: [(300.0, 0)], 40: [(50.0, 1)],
                                                                                    50: [(60.0, 2)]})
    assert m.st["n_atrapadas"] == 1 and m.st["reemplazos_deposito"] == 0, m.st
    assert m.st["reemplazos_4"] == 1, "50 no cubre 100; el 4.º slot con 50 + 60 = 110 sí"
    assert abs(m.holding_btc - mm._redondear(100 * (1 - com) / fill)) < 1e-12
    assert abs(m.st["atr_btc_bot"] / m.st["atr_btc_dca"] - 1) < 1e-4, "1.ª compra el día del depósito ≈ DCA"
    assert abs(m.conservacion()) < 1e-9
    print("✔ atrapado: 50 va al 4.º slot, con 60 más llega a 110 y reemplaza al de 100")

    # 3) filtro: depósito en VENTA espera sin operar, el DCA compra ese día, entra en la 1.ª vela de COMPRA
    v = _velas_sinteticas(4, tocar_tp=False)
    v.iloc[48, v.columns.get_loc("open")] = 105.0
    reg = np.arange(len(v)) >= 48
    sen = np.zeros(len(v), bool)
    sin_atr, rg = np.full(len(v), np.nan), np.full(len(v), 0.02)
    m, _ = correr_lotes(v.iloc[:48], sen[:48], sin_atr[:48], rg[:48], {5: [(300.0, 0)]}, regimen=reg[:48])
    assert [r[0] for r in m.reserva_filtro] == [300.0] and all(s.nominal == 0 for s in m.slots)
    assert m.bench_btc == mm._redondear(300 * (1 - com) / (100 * (1 + slip)))
    assert abs(m.conservacion()) < 1e-9 and abs(m.valor_btc(100.0) - 3.0) < 1e-12
    m, _ = correr_lotes(v.iloc[:49], sen[:49], sin_atr[:49], rg[:49], {5: [(300.0, 0)]}, regimen=reg[:49])
    assert not m.reserva_filtro and [round(s.nominal, 9) for s in m.slots[:3]] == [100.0] * 3
    assert all(x["t0"] == v.index[5] and x["p0"] == 100.0 for k in range(3) for x in m.tr[k])
    r = dca_filtro(v, reg, {5: [(300.0, 0)]}, pd.Timedelta(hours=1))
    assert r["bot_btc"] == mm._redondear(300 * (1 - com) / (105 * (1 + slip))), r
    print("✔ filtro: el depósito en VENTA espera y entra en la 1.ª vela de COMPRA; dca_filtro compra ahí a 105")

    # 4) contraste: sin límite de ciclos ni atrapados, un depósito inicial = Motor v0.3 con G4
    _init()
    desde, hasta = PERIODOS["2024-2026"]
    sl = _V.loc[(_V.index >= desde) & (_V.index < hasta)]
    sen_a = _SEN.reindex(sl.index).fillna(False).to_numpy()
    atr_prev = _IND["atr"].shift(1).reindex(sl.index).to_numpy()
    rango_prev = _IND["rango"].shift(1).reindex(sl.index).to_numpy()
    m, _ = correr_lotes(sl, sen_a, atr_prev, rango_prev, {0: [(600.0, 0)]}, max_ciclos=np.inf, atrapar=False)
    cfg = mm.Config(slots="A", salida=SALIDA, ganancia="G4", capital=600.0, cierre_forzado=False)
    r = mm.simular(_V, _SEN, _IND["atr"], cfg, lambda k: 0.0, desde, hasta, rango=_IND["rango"])
    a = pd.DataFrame(m.operaciones)[["slot", "entrada_t", "salida_t", "ganancia"]]
    b = pd.DataFrame(r.motor.operaciones)[["slot", "entrada_t", "salida_t", "ganancia"]]
    assert len(a) == len(b), (len(a), len(b))
    assert (a[["slot", "entrada_t", "salida_t"]].values == b[["slot", "entrada_t", "salida_t"]].values).all()
    assert np.allclose(a["ganancia"], b["ganancia"], atol=1e-9)
    print(f"✔ contraste con el motor actual (v0.3 + G4): {len(a)} operaciones idénticas")

    # 5) el filtro siempre en COMPRA da exactamente lo mismo que sin filtro
    cal = calendario(sl.index, desde, hasta, np.random.default_rng([0, 2]))
    m1, _ = correr_lotes(sl, sen_a, atr_prev, rango_prev, cal["deps"], espera_cuenta=True)
    m2, _ = correr_lotes(sl, sen_a, atr_prev, rango_prev, cal["deps"], espera_cuenta=True,
                         regimen=np.ones(len(sl), bool))
    p = float(sl["close"].iloc[-1])
    assert m1.valor_btc(p) == m2.valor_btc(p) and len(m1.operaciones) == len(m2.operaciones)
    print(f"✔ fb4 con régimen siempre COMPRA = b4 ({len(m1.operaciones)} operaciones)")


# --- informe ------------------------------------------------------------------------
VARIANTES = {"actual": None, "b1": "placebo_b1", "b2": "placebo_b2", "b4": "placebo_b4",
             "fb1": "placebo_r_fb1", "fb2": "placebo_r_fb2", "fb4": "placebo_r_fb4",
             "dca_filtro": "placebo_r_dca_filtro"}


def _q(s: pd.Series) -> str:
    return f"{s.median():+.2f} [{s.quantile(0.1):+.2f}, {s.quantile(0.9):+.2f}]"


def _concentracion(d: pd.DataFrame) -> float:
    """Mediana entre calendarios de: el mayor aumento mensual de ventaja / la suma de los aumentos."""
    conc = []
    for fotos in d["fotos"].dropna():
        f = pd.DataFrame(fotos, columns=["t", "bot", "bench", "precio"])
        dif = (f["bot"] - f["bench"]) * f["precio"]
        inc = dif.diff().fillna(dif)
        pos = inc.clip(lower=0).sum()
        conc.append(inc.max() / pos if pos > 0 else np.nan)
    return float(np.nanmedian(conc)) if conc else np.nan


def informe(df: pd.DataFrame) -> None:
    tramos = ["2021-2023", "2024-2026"]
    filas = []
    for var, plac in VARIANTES.items():
        g = lambda periodo, modo=var: df[(df.periodo == periodo) & (df.modo == modo)]
        cont = g("continuo")
        fila = {"variante": var, "continuo": _q(cont["ventaja_pct"]), "gana_cont": (cont["ventaja_pct"] > 0).mean()}
        ok1 = ok2 = True
        tr = tramos + (["d_2017-2020"] if var == "dca_filtro" else [])
        for t in tr:
            d = g(t)
            fila[t] = f"{d['ventaja_pct'].median():+.2f} ({(d['ventaja_pct'] > 0).mean():.0%})"
            ok1 &= d["ventaja_pct"].median() > 0
            ok2 &= (d["ventaja_pct"] > 0).mean() >= 0.60
        if plac:
            p90 = g("continuo", plac)["ventaja_pct"].quantile(0.9)
            fila["placebo_p90"] = f"{p90:+.2f}"
            ok3 = cont["ventaja_pct"].median() > p90
        else:
            fila["placebo_p90"], ok3 = "—", None
        conc = _concentracion(cont) if "fotos" in cont else np.nan
        fila["mejor_mes"] = f"{conc:.0%}" if not np.isnan(conc) else "—"
        ok4 = None if np.isnan(conc) else conc <= 0.5
        fila["criterios"] = "".join("✔" if c else ("—" if c is None else "✘") for c in (ok1, ok2, ok3, ok4))
        fila["PASA"] = "SÍ" if all(c is not False for c in (ok1, ok2, ok3, ok4)) and ok3 else "no"
        filas.append(fila)
    print("\nVentaja de BTC final contra el DCA puro, % (mediana [p10, p90]; entre paréntesis el % de calendarios que gana)")
    print("Criterios: (1) mediana > 0 en cada tramo · (2) gana ≥ 60 % en cada tramo · (3) > p90 del placebo · "
          "(4) mejor mes ≤ 50 %")
    print(pd.DataFrame(filas).set_index("variante").to_string())

    print("\nDetalle del continuo 2021-2026 (medianas):")
    for var in VARIANTES:
        d = df[(df.periodo == "continuo") & (df.modo == var)]
        if var == "dca_filtro":
            print(f"  {var:10} USDT esperando {d['ocioso_pct'].median():.0f} % del tiempo-dinero · "
                  f"sin comprar al final {d['usdt_espera_fin'].median():,.0f} USDT")
            continue
        neto = d["realizado"] - d["atrapadas_costo"] - d["latente_fin"]
        linea = (f"  {var:10} trading neto {neto.median():+,.0f} USDT (realizada {d['realizado'].median():,.0f} − "
                 f"atrapadas {d['atrapadas_costo'].median():,.0f}) · operaciones {d['operaciones'].median():.0f}")
        if var != "actual":
            linea += (f" · ocioso {d['ocioso_pct'].median():.0f} % · atrapadas contra el día del depósito "
                      f"{d['atr_vs_dep_pct'].median():+.1f} % de BTC ({d['atr_usdt'].median():,.0f} USDT)")
        print(linea)
    d = df[(df.modo == "dca_filtro") & df.periodo.str.startswith("d_")]
    print("\nFiltro sin bot con velas diarias (mediana, % de calendarios que gana):")
    for periodo, g in d.groupby("periodo", sort=False):
        print(f"  {periodo}: {_q(g['ventaja_pct'])} · gana en {(g['ventaja_pct'] > 0).mean():.0%}")
    p = df[(df.modo == "placebo_r_dca_filtro") & (df.periodo == "d_continuo")]["ventaja_pct"]
    if len(p):
        print(f"  placebo d_continuo: {_q(p)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=8)
    ap.add_argument("--semillas", type=int, default=200)
    ap.add_argument("--prueba", action="store_true")
    args = ap.parse_args()
    if args.prueba:
        prueba()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_plac = min(N_PLACEBO, args.semillas)
    tareas = [(p, s, m) for p in PERIODOS for s in range(args.semillas) for m in VARIANTES]
    tareas += [(p, s, pl) for p in PERIODOS for s in range(n_plac) for pl in VARIANTES.values() if pl]
    tareas += [(p, s, "dca_filtro") for p in PERIODOS_D for s in range(args.semillas)]
    tareas += [(p, s, "placebo_r_dca_filtro") for p in PERIODOS_D for s in range(n_plac)]
    with Pool(args.procesos, initializer=_init) as pool:
        df = pd.DataFrame(pool.map(correr, tareas, chunksize=4))
    df.drop(columns="fotos").to_csv(OUT / "mejoras.csv", index=False)
    print(f"{len(tareas)} corridas en {time.time() - t0:.0f} s → {OUT / 'mejoras.csv'}")
    informe(df)


if __name__ == "__main__":
    main()
