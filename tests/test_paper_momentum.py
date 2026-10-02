"""Prueba hacia adelante sin dinero del momentum entre pares (02/10/2026)."""
import json

import pandas as pd
import pytest

import pkg.paper_momentum as pm

T0 = pd.Timestamp("2026-10-03 00:30", tz="UTC")
SYMS = [f"S{i:02d}-USDT" for i in range(12)]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    cfg = {"enabled": True, "lookback_h": 3, "hold_h": 6, "n_largos": 2, "n_cortos": 2,
           "costo_por_vuelta": 0.0016, "universo": SYMS}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(pm, "CONFIG_PATH", p)
    monkeypatch.setattr(pm, "OUT_DIR", tmp_path / "paper")
    monkeypatch.setattr(pm.time, "sleep", lambda *_: None)
    eventos = []
    monkeypatch.setattr(pm, "_avisar", lambda r, f: eventos.append(r))
    return tmp_path, eventos


def fake_fetch(sym, limit, ahora):
    """Precio que sube (k−5)% por hora: S11 la que más sube, S00 la que más baja."""
    k = int(sym[1:3]) - 5
    horas = pd.date_range(end=ahora.floor("1h") - pd.Timedelta(hours=1), periods=limit, freq="1h")
    pasos = (horas - pd.Timestamp("2026-09-01", tz="UTC")) / pd.Timedelta(hours=1)   # tiempo absoluto
    return pd.Series([100 * (1 + k / 100) ** n for n in pasos], index=horas)


def test_primer_rebalanceo_arma_el_libro(entorno):
    tmp, eventos = entorno
    r = pm.run_paper_momentum(ahora=T0, fetch=fake_fetch)
    libro = pd.read_csv(tmp / "paper/momentum_libro.csv")
    assert set(libro[libro.lado == 1].symbol) == {"S10-USDT", "S11-USDT"}
    assert set(libro[libro.lado == -1].symbol) == {"S00-USDT", "S01-USDT"}
    assert not (tmp / "paper/momentum_trades.csv").exists()
    estado = json.loads((tmp / "paper/momentum_estado.json").read_text())
    assert estado["rebalanceos"] == 1
    assert pd.Timestamp(estado["proximo"]) == T0.floor("1h") + pd.Timedelta(hours=6)
    assert r["rebalanceo"] == 1 and len(eventos) == 1


def test_no_hace_nada_si_no_toca(entorno):
    pm.run_paper_momentum(ahora=T0, fetch=fake_fetch)
    llamadas = []
    assert pm.run_paper_momentum(ahora=T0 + pd.Timedelta(hours=1),
                                 fetch=lambda *a: llamadas.append(a)) is None
    assert llamadas == []


def test_segundo_rebalanceo_cierra_con_el_pnl_correcto(entorno):
    tmp, _ = entorno
    pm.run_paper_momentum(ahora=T0, fetch=fake_fetch)
    r = pm.run_paper_momentum(ahora=T0 + pd.Timedelta(hours=6), fetch=fake_fetch)
    t = pd.read_csv(tmp / "paper/momentum_trades.csv")
    assert len(t) == 4
    # largos que subían y cortos que bajaban: las cuatro patas ganan en bruto
    assert (t.bruto > 0).all()
    s11 = t[t.symbol == "S11-USDT"].iloc[0]
    assert s11.bruto == pytest.approx(1.06 ** 6 - 1, rel=1e-9)
    assert s11.neto == pytest.approx(s11.bruto - 0.0016)
    s00 = t[t.symbol == "S00-USDT"].iloc[0]
    assert s00.bruto == pytest.approx(-(0.95 ** 6 - 1), rel=1e-9)
    assert r["rebalanceo"] == 2 and r["patas_cerradas"] == 4


def test_un_par_sin_precio_no_frena_el_rebalanceo(entorno):
    def fetch(sym, limit, ahora):
        if sym == "S05-USDT":
            raise RuntimeError("API caída")
        return fake_fetch(sym, limit, ahora)
    assert pm.run_paper_momentum(ahora=T0, fetch=fetch)["universo_con_precio"] == 11


def test_universo_insuficiente_no_rebalancea(entorno):
    assert pm.run_paper_momentum(ahora=T0, fetch=lambda *a: pd.Series(dtype=float)) is None


def test_nunca_levanta_excepcion(entorno, monkeypatch):
    tmp, _ = entorno
    (tmp / "cfg.json").write_text("{roto")
    assert pm.run_paper_momentum(ahora=T0, fetch=fake_fetch) is None


def test_deshabilitado_no_hace_nada(entorno):
    tmp, _ = entorno
    c = json.loads((tmp / "cfg.json").read_text()); c["enabled"] = False
    (tmp / "cfg.json").write_text(json.dumps(c))
    assert pm.run_paper_momentum(ahora=T0, fetch=fake_fetch) is None


def test_la_hora_en_formacion_no_se_usa(monkeypatch):
    import pkg.price_bingx_5m as px
    ahora = pd.Timestamp("2026-10-03 10:20", tz="UTC")
    velas = [dict(symbol="X", open=1, high=1, low=1, close=float(h), volume=1,
                  date=pd.Timestamp("2026-10-03", tz="UTC") + pd.Timedelta(hours=h)) for h in range(11)]
    monkeypatch.setattr(px, "_fetch_bingx_candles", lambda *a, **k: velas)
    s = pm._precios_1h("X", 20, ahora)
    assert s.index.max() == pd.Timestamp("2026-10-03 09:00", tz="UTC")   # la de 10:00 está en formación


def test_la_config_real_es_valida():
    cfg = json.loads(pm.CONFIG_PATH.read_text())
    assert len(cfg["universo"]) == 39 and cfg["lookback_h"] == 168 and cfg["hold_h"] == 72
    assert cfg["n_largos"] == 5 and cfg["n_cortos"] == 5
