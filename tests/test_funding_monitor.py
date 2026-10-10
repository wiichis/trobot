"""Monitor del carry de funding (09/10/2026)."""
import pandas as pd
import pytest

import pkg.funding_monitor as fm

AHORA = pd.Timestamp("2026-10-09 02:00", tz="UTC")
CFG = {"umbral_alto": 0.20, "umbral_salida": 0.15, "umbral_medio": 0.10}


def eventos(anual, dias=40, cada_h=8):
    """Eventos con una tasa constante que anualiza a `anual`."""
    por_ev = anual / 365 / (24 / cada_h)
    fin = int(AHORA.timestamp() * 1000)
    return [{"ms": fin - k * cada_h * 3600_000, "tasa": por_ev} for k in range(dias * 24 // cada_h)][::-1]


def fuentes(tabla):
    return {"tickers": lambda: [{"symbol": s, "quoteVolume": "9000000"} for s in tabla],
            "spot": lambda: {s: 500_000.0 for s in tabla}, "funding": lambda s: tabla[s]}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "OUT_DIR", tmp_path)
    avisos = []
    monkeypatch.setattr(fm, "_emitir", lambda **kw: avisos.append(kw))
    return avisos


def mercado(nivel_top):
    t = {f"A{i}-USDT": eventos(0.05) for i in range(25)}
    t.update({f"T{i}-USDT": eventos(nivel_top) for i in range(10)})
    return t


def test_funding_anualizado_respeta_ventana_e_historia_corta():
    assert fm.funding_anualizado(eventos(0.30), int(AHORA.timestamp() * 1000), 30) == pytest.approx(0.30, rel=0.02)
    assert fm.funding_anualizado(eventos(0.30, dias=10), int(AHORA.timestamp() * 1000), 30) is None
    assert fm.funding_anualizado(eventos(0.30, cada_h=1), int(AHORA.timestamp() * 1000), 30) == pytest.approx(0.30, rel=0.02)


def test_histeresis_del_nivel():
    assert fm.nivel(0.21, None, CFG) == "alto"
    assert fm.nivel(0.17, "alto", CFG) == "alto"      # no sale hasta bajar de 15%
    assert fm.nivel(0.17, "medio", CFG) == "medio"    # no entra hasta llegar a 20%
    assert fm.nivel(0.14, "alto", CFG) == "medio"
    assert fm.nivel(0.05, "medio", CFG) == "bajo"


def test_arranque_avisa_y_no_repite_el_mismo_dia(entorno):
    r = fm.run_funding_monitor(ahora=AHORA, fuentes=fuentes(mercado(0.13)))
    assert r["nivel"] == "medio" and len(entorno) == 1 and "Estado actual" in entorno[0]["mensaje_md"]
    assert fm.run_funding_monitor(ahora=AHORA, fuentes=fuentes(mercado(0.13))) == {}
    assert len(entorno) == 1


def test_avisa_solo_al_entrar_y_salir_de_alto(entorno):
    dia = lambda k: AHORA + pd.Timedelta(days=k)
    fm.run_funding_monitor(ahora=dia(0), fuentes=fuentes(mercado(0.13)))             # arranque
    assert fm.run_funding_monitor(ahora=dia(1), fuentes=fuentes(mercado(0.12))) == {}  # sigue medio: sin aviso
    r = fm.run_funding_monitor(ahora=dia(2), fuentes=fuentes(mercado(0.30)))         # entra a alto
    assert r["nivel"] == "alto" and "ALTO" in entorno[-1]["mensaje_md"]
    assert fm.run_funding_monitor(ahora=dia(3), fuentes=fuentes(mercado(0.17))) == {}  # histéresis
    r = fm.run_funding_monitor(ahora=dia(4), fuentes=fuentes(mercado(0.08)))         # sale
    assert "salió" in entorno[-1]["mensaje_md"] and len(entorno) == 3


def test_pocos_pares_o_error_no_escribe_estado_ni_levanta(entorno):
    assert fm.run_funding_monitor(ahora=AHORA, fuentes=fuentes({"A-USDT": eventos(0.3)})) is None
    assert fm.leer_estado() is None
    def roto(): raise RuntimeError("api caída")
    assert fm.run_funding_monitor(ahora=AHORA, fuentes={"tickers": roto, "spot": roto, "funding": roto}) is None


def test_no_corre_antes_de_la_1am_utc(entorno):
    assert fm.run_funding_monitor(ahora=AHORA.replace(hour=0, minute=30), fuentes=fuentes(mercado(0.3))) is None


def test_excluye_pares_con_contado_fino(entorno):
    t = mercado(0.13)
    t["FINO-USDT"] = eventos(0.80)
    f = fuentes(t)
    f["spot"] = lambda: {**{s: 500_000.0 for s in t}, "FINO-USDT": 40_000.0}
    r = fm.run_funding_monitor(ahora=AHORA, fuentes=f)
    assert r["nivel"] == "medio" and all(x["par"] != "FINO-USDT" for x in fm.leer_estado()["top"])
