"""Bot BTC por rangos en modo real (24/10/2026) contra un BingX spot falso.

El exchange falso responde con la forma real de BingX spot (ids numéricos, cantidades como
texto, fee negativo y en BTC en la compra, en USDT en la venta) y llena las órdenes límite de
venta cuando el máximo de la vela llega al precio, que es la regla del backtest.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
import requests

import pkg.btc_rangos as papel
import pkg.btc_rangos_real as br
from pkg import btc_rangos_motor as mm
from pkg.bingx_spot import ClienteSpot

CFG = json.loads(br.CONFIG_PATH.read_text(encoding="utf-8"))
STEP = 1e-6


def serie(dias=75, semilla=7):
    idx = pd.date_range("2024-01-01", periods=dias * 24, freq="h")
    rng = np.random.default_rng(semilla)
    c = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.006, len(idx))))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, len(idx)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, len(idx)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)


class BingxSpotFalso:
    def __init__(self, usdt=186.66, slippage=0.0005, fee=0.001):
        self.usdt, self.usdt_lock, self.btc, self.btc_lock = usdt, 0.0, 0.0, 0.0
        self.precio = 40_000.0
        self.slip, self.fee = slippage, fee
        self.ordenes = {}
        self.sig_id = 1_850_000_000_000_000_000
        self.rechazar = None      # code a devolver en la próxima orden
        self.timeout_tras_crear = False
        self.llenado_parcial = None  # fracción a llenar en la próxima vela

    # --- simulación del mercado ---
    def jugar_vela(self, high: float) -> None:
        for o in self.ordenes.values():
            if o["side"] == "SELL" and o["status"] in ("NEW", "PARTIALLY_FILLED") and high >= o["price"]:
                frac = self.llenado_parcial or 1.0
                self.llenado_parcial = None
                self._llenar_venta(o, o["price"], o["origQty"] * frac - o["executedQty"])

    def _llenar_venta(self, o, precio, qty):
        qty = round(qty, 6)
        quote = qty * precio
        fee = quote * self.fee
        o["executedQty"] += qty
        o["cummulativeQuoteQty"] += quote
        o["fee"] += fee
        o["feeAsset"] = "USDT"
        self.btc_lock -= qty
        self.usdt += quote - fee
        o["status"] = "FILLED" if o["executedQty"] >= o["origQty"] - 1e-12 else "PARTIALLY_FILLED"

    # --- API ---
    def __call__(self, metodo, path, params, firmado):
        ok = lambda data: {"code": 0, "msg": "", "data": data}
        if path.endswith("common/symbols"):
            return ok({"symbols": [{"symbol": "BTC-USDT", "minQty": 0.0001826, "maxQty": 18.2, "minNotional": 0.5,
                                    "tickSize": 0.01, "stepSize": 1e-06, "apiStateSell": True, "apiStateBuy": True}]})
        if path.endswith("account/balance"):
            return ok({"balances": [{"asset": "USDT", "free": f"{self.usdt:.8f}", "locked": f"{self.usdt_lock:.8f}"},
                                    {"asset": "BTC", "free": f"{self.btc:.8f}", "locked": f"{self.btc_lock:.8f}"}]})
        if path.endswith("ticker/price"):
            return ok([{"symbol": "BTC_USDT", "trades": [{"price": f"{self.precio:.2f}", "volume": "0.01"}]}])
        if path.endswith("ticker/bookTicker"):
            return ok([{"symbol": "BTC-USDT", "bidPrice": f"{self.precio:.2f}", "askPrice": f"{self.precio + 0.01:.2f}"}])
        if path.endswith("user/commissionRate"):
            return ok({"takerCommissionRate": self.fee, "makerCommissionRate": self.fee})
        if path.endswith("trade/order"):
            if self.rechazar:
                code, self.rechazar = self.rechazar, None
                return {"code": code, "msg": "order rejected", "data": {}}
            oid = self.sig_id
            self.sig_id += 7
            o = {"orderId": oid, "clientOrderID": params["newClientOrderId"], "side": params["side"],
                 "type": params["type"], "price": float(params.get("price", 0)), "origQty": 0.0,
                 "executedQty": 0.0, "cummulativeQuoteQty": 0.0, "fee": 0.0, "feeAsset": "", "status": "NEW"}
            if params["type"] == "MARKET":
                usdt = float(params["quoteOrderQty"])
                assert usdt <= self.usdt + 1e-9, "compra sin saldo"
                fill = self.precio * (1 + self.slip)
                qty = math.floor(usdt / fill / STEP + 1e-9) * STEP
                o.update(origQty=qty, executedQty=qty, cummulativeQuoteQty=qty * fill, fee=qty * self.fee,
                         feeAsset="BTC", status="FILLED")
                self.usdt -= qty * fill
                self.btc += qty * (1 - self.fee)
            else:
                qty = float(params["quantity"])
                assert abs(qty / STEP - round(qty / STEP)) < 1e-6, "cantidad fuera del step"
                assert abs(o["price"] / 0.01 - round(o["price"] / 0.01)) < 1e-6, "precio fuera del tick"
                assert qty <= self.btc + 1e-12, "venta sin BTC"
                o["origQty"] = qty
                self.btc -= qty
                self.btc_lock += qty
                if o["price"] <= self.precio:  # límite bajo el mercado: se llena ya, al mercado
                    self._llenar_venta(o, self.precio, qty)
            self.ordenes[oid] = o
            if self.timeout_tras_crear:
                self.timeout_tras_crear = False
                raise requests.exceptions.ReadTimeout("timeout")
            return ok(self._json(o))
        if path.endswith("trade/query"):
            if "orderId" in params:
                o = self.ordenes.get(int(params["orderId"]))
            else:
                o = next((x for x in self.ordenes.values() if x["clientOrderID"] == params["clientOrderID"]), None)
            if o is None:
                return {"code": 100404, "msg": "order not exist", "data": {}}
            return ok(self._json(o))
        if path.endswith("trade/cancel"):
            o = self.ordenes[int(params["orderId"])]
            if o["status"] in ("NEW", "PARTIALLY_FILLED"):
                resto = o["origQty"] - o["executedQty"]
                self.btc_lock -= resto
                self.btc += resto
                o["status"] = "CANCELED"
            return ok(self._json(o))
        raise AssertionError(f"endpoint no previsto {path}")

    @staticmethod
    def _json(o):
        return {"symbol": "BTC-USDT", "orderId": o["orderId"], "clientOrderID": o["clientOrderID"],
                "price": f"{o['price']:.2f}", "origQty": f"{o['origQty']:.6f}",
                "executedQty": f"{o['executedQty']:.6f}", "cummulativeQuoteQty": f"{o['cummulativeQuoteQty']:.8f}",
                "status": o["status"], "type": o["type"], "side": o["side"], "fee": f"{-o['fee']:.10f}",
                "feeAsset": o["feeAsset"], "time": 1, "updateTime": 1, "origQuoteOrderQty": "0"}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    cfg = dict(CFG, enabled=True, modo="real")
    f = tmp_path / "btc_rangos_real.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(br, "CONFIG_PATH", f)
    monkeypatch.setattr(br, "OUT_DIR", tmp_path / "real")
    monkeypatch.setattr(br, "_AVISADO", set())
    monkeypatch.setattr(br, "_ERRORES", {"seguidos": 0})
    monkeypatch.setattr("pkg.bingx_spot.time.sleep", lambda s: None)
    avisos = []
    monkeypatch.setattr(br, "_emitir", lambda cat, sev, **kw: avisos.append((cat, sev, kw)))
    ex = BingxSpotFalso()
    return {"cfg": cfg, "cfg_path": f, "avisos": avisos, "dir": tmp_path / "real", "ex": ex,
            "cli": ClienteSpot(transporte=ex)}


class Reloj:
    """Corre el job hora a hora: antes de cada corrida el exchange 'juega' la vela que cerró."""

    def __init__(self, entorno, full, k):
        self.e, self.full, self.k = entorno, full, k

    def correr(self):
        idx = self.full.index
        ahora = (idx[self.k + 1] + pd.Timedelta(minutes=1)).tz_localize("UTC")
        self.e["ex"].precio = float(self.full["open"].iloc[self.k + 1])
        vis = self.full[self.full.index <= ahora.tz_localize(None)]
        return br.run_btc_rangos_real(ahora, self.e["cli"], lambda limit: vis.tail(limit))

    def avanzar(self):
        self.k += 1
        self.e["ex"].jugar_vela(float(self.full["high"].iloc[self.k]))


def test_arranque_exige_capital_en_spot(entorno):
    entorno["ex"].usdt = 50.0
    full = serie()
    r = Reloj(entorno, full, 40 * 24)
    assert r.correr() is None
    assert not (entorno["dir"] / "estado.json").exists()
    assert entorno["avisos"][-1][0] == "btc_real_alerta"
    entorno["ex"].usdt = 186.66
    assert r.correr()["nuevo"]
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["usdt_ajeno"] == pytest.approx(6.66)
    assert entorno["avisos"][-1][0] == "btc_real_inicio"


def test_decisiones_iguales_al_backtest_y_cuentas_cuadradas(entorno):
    """Mismas compras y ventas (slot, hora de entrada, hora de salida) que `mm.simular` sobre las
    mismas velas, cruzando un cierre mensual con DCA depositado en spot. Y la contabilidad del
    bot cuadra con los saldos del exchange."""
    full = serie()
    ex = entorno["ex"]
    k0 = int(np.searchsorted(full.index, pd.Timestamp("2024-02-10 00:00"))) - 1  # última cerrada: 09/02 23:00
    reloj = Reloj(entorno, full, k0)
    assert reloj.correr()["nuevo"]
    fin = pd.Timestamp("2024-03-15 00:00")
    while full.index[reloj.k + 1] < fin:
        reloj.avanzar()
        if full.index[reloj.k + 1] == pd.Timestamp("2024-03-03 00:00"):
            ex.usdt += 400.0  # el usuario deposita el DCA antes del cierre
        res = reloj.correr()
        assert res is not None and not res["pausa"], entorno["avisos"][-1]
    datos = papel.preparar(full, entorno["cfg"])
    cfg_m = papel.config_motor(entorno["cfg"])
    sim = mm.simular(full, datos["senal"], datos["atr_prev"].shift(-1), cfg_m, lambda k: 400.0,
                     pd.Timestamp("2024-02-10 00:00"), full.index[reloj.k + 1], rango=datos["rango"])
    s_ops = pd.DataFrame(sim.motor.operaciones)
    r_ops = pd.read_csv(entorno["dir"] / "operaciones.csv", parse_dates=["entrada_t", "salida_t"])
    assert len(s_ops) > 5
    # la venta real se registra en la corrida siguiente a la vela en que se llenó (+1 h)
    abiertas_sim = s_ops[s_ops["salida_t"] <= full.index[reloj.k]]
    assert len(r_ops) == len(abiertas_sim)
    assert (r_ops["slot"].to_numpy() == abiertas_sim["slot"].to_numpy()).all()
    assert (r_ops["entrada_t"].to_numpy() == abiertas_sim["entrada_t"].to_numpy()).all()
    # (si la vela abre sobre el TP, la orden se llena al colocarla: misma hora que el backtest)
    assert (r_ops["salida_t"] - abiertas_sim["salida_t"].to_numpy()).isin([pd.Timedelta(0), pd.Timedelta(hours=1)]).all()
    assert np.allclose(r_ops["ganancia"], abiertas_sim["ganancia"], rtol=0.05, atol=0.02)

    est = json.loads((entorno["dir"] / "estado.json").read_text())
    m = papel.motor_desde_dict(est["motor"], cfg_m, lambda k: 0.0)
    assert m.n_cierres == 1
    mensual = pd.read_csv(entorno["dir"] / "mensual.csv")
    assert mensual["deposito_detectado"].iloc[0] == pytest.approx(400.0, abs=0.01)
    assert mensual["compra_holding_btc"].iloc[0] > 0
    # cuentas: USDT del exchange = USDT del bot + reservado para holding + ajeno
    usdt_ex = ex.usdt + ex.usdt_lock
    assert usdt_ex == pytest.approx(m.usdt_bot() + m.usdt_para_holding + est["usdt_ajeno"], abs=0.05)
    # BTC del exchange = slots abiertos + holding (incluye el polvo bajo el step)
    assert ex.btc + ex.btc_lock == pytest.approx(m.btc_en_slots() + m.holding_btc, abs=1e-6)
    # cada slot abierto tiene su orden de venta en el exchange, por su cantidad exacta
    for i, s in enumerate(m.slots, 1):
        if s.abierto:
            o = ex.ordenes[int(est["ordenes"][str(i)]["order_id"])]
            assert o["status"] == "NEW" and o["origQty"] == pytest.approx(s.qty)
            assert o["price"] >= m.piso(s)
    cats = [a[0] for a in entorno["avisos"]]
    assert "btc_real_operaciones" in cats and "btc_real_cierre_mensual" in cats and "btc_real_resumen" in cats
    assert "btc_real_recordatorio_dca" in cats
    assert not any(a[1] == "CRITICAL" for a in entorno["avisos"])


def _hasta_primera_compra(entorno, full, reloj, limite=400):
    for _ in range(limite):
        reloj.avanzar()
        reloj.correr()
        est = json.loads((entorno["dir"] / "estado.json").read_text())
        if est["ordenes"]:
            return est
    raise AssertionError("no hubo compra")


def test_timeout_al_comprar_se_recupera_por_client_id(entorno):
    full = serie()
    ex = entorno["ex"]
    reloj = Reloj(entorno, full, 40 * 24)
    reloj.correr()
    # avanzar hasta la hora anterior a una compra, y que esa compra dé timeout
    for _ in range(400):
        reloj.avanzar()
        ex.timeout_tras_crear = True
        n = len(ex.ordenes)
        assert reloj.correr() is None or len(ex.ordenes) == n
        if len(ex.ordenes) > n:
            break
        ex.timeout_tras_crear = False
    else:
        raise AssertionError("no hubo compra")
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["compra_en_curso"] and not est["ordenes"]  # la orden existe pero el bot no la vio
    assert entorno["avisos"][-1][1] == "CRITICAL"
    reloj.avanzar()
    reloj.correr()
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["compra_en_curso"] is None
    m = papel.motor_desde_dict(est["motor"], papel.config_motor(entorno["cfg"]), lambda k: 0.0)
    assert sum(s.abierto for s in m.slots) == 1 and est["ordenes"]
    assert ex.btc + ex.btc_lock == pytest.approx(m.btc_en_slots() + m.holding_btc, abs=1e-6)


def test_orden_rechazada_no_abre_el_slot(entorno):
    full = serie()
    ex = entorno["ex"]
    reloj = Reloj(entorno, full, 40 * 24)
    reloj.correr()
    for _ in range(400):
        reloj.avanzar()
        ex.rechazar = 100400
        reloj.correr()
        if ex.rechazar is None:
            break
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["compra_en_curso"] is None and not est["ordenes"]
    m = papel.motor_desde_dict(est["motor"], papel.config_motor(entorno["cfg"]), lambda k: 0.0)
    assert not any(s.abierto for s in m.slots)
    assert entorno["avisos"][-1][1] == "CRITICAL"


def test_btc_que_falta_en_la_cuenta_pausa_las_compras(entorno):
    full = serie()
    ex = entorno["ex"]
    reloj = Reloj(entorno, full, 40 * 24)
    reloj.correr()
    _hasta_primera_compra(entorno, full, reloj)
    ex.btc_lock -= 0.0005  # alguien vendió o retiró BTC que era del bot
    reloj.avanzar()
    reloj.correr()
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["pausa"] and "BTC" in est["pausa"]["motivo"]
    n = len(ex.ordenes)
    for _ in range(72):
        reloj.avanzar()
        reloj.correr()
    compras = [o for o in list(ex.ordenes.values())[n:] if o["side"] == "BUY"]
    assert not compras


def test_llenado_parcial_al_cierre_mensual(entorno):
    """La venta del slot se llena en parte y llega el cierre: la parte vendida se realiza y el resto
    pasa a holding."""
    full = serie()
    ex = entorno["ex"]
    k = int(np.searchsorted(full.index, pd.Timestamp("2024-03-02 06:00")))
    reloj = Reloj(entorno, full, k)
    reloj.correr()
    est = _hasta_primera_compra(entorno, full, reloj, limite=17)
    oid = int(est["ordenes"]["1"]["order_id"]) if "1" in est["ordenes"] else int(next(iter(est["ordenes"].values()))["order_id"])
    o = ex.ordenes[oid]
    ex._llenar_venta(o, o["price"], round(o["origQty"] / 2, 6))
    assert o["status"] == "PARTIALLY_FILLED"
    while full.index[reloj.k + 1] < pd.Timestamp("2024-03-03 00:00"):
        reloj.avanzar()
        reloj.correr()
        if o["status"] == "FILLED":
            pytest.skip("la serie llenó la orden entera antes del cierre")
    reloj.avanzar()
    reloj.correr()
    ops = pd.read_csv(entorno["dir"] / "operaciones.csv")
    assert (ops["motivo"] == "tp_parcial").sum() == 1
    mensual = pd.read_csv(entorno["dir"] / "mensual.csv")
    assert mensual["n_atrapadas"].iloc[0] >= 1
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    m = papel.motor_desde_dict(est["motor"], papel.config_motor(entorno["cfg"]), lambda k: 0.0)
    assert ex.btc + ex.btc_lock == pytest.approx(m.btc_en_slots() + m.holding_btc, abs=1e-6)


def test_estado_ilegible_no_opera(entorno):
    full = serie()
    reloj = Reloj(entorno, full, 40 * 24)
    reloj.correr()
    f = entorno["dir"] / "estado.json"
    f.write_text("{roto", encoding="utf-8")
    n = len(entorno["ex"].ordenes)
    for _ in range(48):
        reloj.avanzar()
        assert reloj.correr() is None
    assert f.read_text(encoding="utf-8") == "{roto"
    assert len(entorno["ex"].ordenes) == n
    assert entorno["avisos"][-1][:2] == ("btc_real_alerta", "CRITICAL")


def test_modo_papel_en_la_config_real_no_corre(entorno):
    entorno["cfg_path"].write_text(json.dumps(dict(entorno["cfg"], modo="papel")), encoding="utf-8")
    full = serie()
    assert Reloj(entorno, full, 40 * 24).correr() is None
    assert not (entorno["dir"] / "estado.json").exists()


def test_redondeo_al_step_y_al_tick():
    from pkg.bingx_spot import Reglas
    r = Reglas("BTC-USDT", 1e-6, 0.01, 0.0001826, 0.5, True, True)
    assert r.qty_abajo(0.0007499999) == "0.000749"
    assert r.qty_abajo(0.000750) == "0.000750"
    assert r.precio_arriba(81234.561) == "81234.57"
    assert r.precio_arriba(81234.56) == "81234.56"


def test_compra_que_nunca_llego_al_exchange_se_descarta(entorno):
    """Corte de red ANTES de que BingX reciba la orden: no existe (100404) y se limpia sin pausar."""
    full = serie()
    ex = entorno["ex"]
    reloj = Reloj(entorno, full, 40 * 24)
    reloj.correr()
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    est["compra_en_curso"] = {"slot": 1, "client_id": "BRB1NUNCA", "t": "2024-02-10 00:00:00", "usdt": "60.00",
                              "motivo": "compra"}
    (entorno["dir"] / "estado.json").write_text(json.dumps(est), encoding="utf-8")
    reloj.avanzar()
    reloj.correr()
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    assert est["compra_en_curso"] is None and not est["pausa"]


def test_cliente_sin_clave_spot_no_usa_la_de_perpetuos(monkeypatch):
    from pkg.bingx_spot import ErrorExchange
    monkeypatch.delenv("TROBOT_BINGX_SPOT_APIKEY", raising=False)
    monkeypatch.setenv("TROBOT_BINGX_APIKEY", "clave-de-perpetuos")
    with pytest.raises(ErrorExchange):
        ClienteSpot.desde_credenciales()


def test_si_falla_la_compra_de_holding_el_cierre_no_se_repite(entorno):
    full = serie()
    ex = entorno["ex"]
    k = int(np.searchsorted(full.index, pd.Timestamp("2024-03-02 20:00")))
    reloj = Reloj(entorno, full, k)
    reloj.correr()
    while full.index[reloj.k + 2] < pd.Timestamp("2024-03-03 00:00"):
        reloj.avanzar()
        reloj.correr()
    reloj.avanzar()            # la corrida que sigue es la del cierre (vela que abre el 03/03 00:00)
    ex.usdt += 400.0
    ex.rechazar = 100400       # la primera orden del cierre es la compra de holding
    assert reloj.correr() is None
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    m = papel.motor_desde_dict(est["motor"], papel.config_motor(entorno["cfg"]), lambda k: 0.0)
    assert m.n_cierres == 1 and m.ciclo == "2024-03"
    assert m.usdt_para_holding > 300   # el DCA quedó reservado en USDT para el próximo cierre
    for _ in range(3):
        reloj.avanzar()
        reloj.correr()
    est = json.loads((entorno["dir"] / "estado.json").read_text())
    m = papel.motor_desde_dict(est["motor"], papel.config_motor(entorno["cfg"]), lambda k: 0.0)
    assert m.n_cierres == 1
    mensual = pd.read_csv(entorno["dir"] / "mensual.csv")
    assert len(mensual) == 1 and mensual["deposito_detectado"].iloc[0] == pytest.approx(400.0, abs=0.01)
    assert ex.usdt + ex.usdt_lock == pytest.approx(m.usdt_bot() + m.usdt_para_holding + est["usdt_ajeno"], abs=0.05)
