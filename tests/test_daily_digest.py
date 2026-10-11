"""Resumen diario de BTC y bolsa (10/10/2026)."""
import datetime as dt
import json

import pytest

import pkg.daily_digest as dd

AHORA = dt.datetime(2026, 10, 12, 16, 30, tzinfo=dt.timezone.utc)      # lunes, 11:30 Lima


def rss(items):
    cuerpo = "".join(f"<item><title>{t}</title><link>{u}</link><pubDate>{d}</pubDate></item>" for t, u, d in items)
    return f"<rss><channel>{cuerpo}</channel></rss>".encode()


def fecha(horas):
    return (AHORA - dt.timedelta(hours=horas)).strftime("%a, %d %b %Y %H:%M:%S +0000")


def test_parse_rss_limpia_utm_y_descarta_incompletos():
    xml = rss([("Bitcoin sube", "https://x.com/a?utm_source=rss&amp;utm_medium=rss", fecha(1)),
               ("", "https://x.com/b", fecha(1)), ("Sin link", "", fecha(1)), ("Sin fecha", "https://x.com/c", "")])
    its = dd.parse_rss(xml, "X")
    assert [i["titulo"] for i in its] == ["Bitcoin sube"] and its[0]["url"] == "https://x.com/a"
    assert dd.parse_rss(b"<<no xml", "X") == []


def test_elegir_es_estricto_recente_y_sin_duplicados():
    xml = rss([("Bitcoin ETF inflows jump as BTC rallies", "https://x/1", fecha(2)),
               ("Bitcoin ETF inflows jump as BTC rallies again", "https://y/1", fecha(3)),     # casi duplicado
               ("Receta de pasta", "https://x/2", fecha(1)),                                   # sin palabras clave
               ("Bitcoin vieja noticia", "https://x/3", fecha(200)),                            # fuera de ventana
               ("SEC aprueba algo", "https://x/4", fecha(5))])
    out = dd.elegir(dd.parse_rss(xml, "X"), ["bitcoin", "btc", "etf", "sec "], AHORA, 30, 5)
    assert [o["url"] for o in out] == ["https://x/1", "https://x/4"]


def test_mensaje_escapa_markdown_y_respeta_limite():
    its = [{"titulo": "A_b *c* [d] " + "x" * 400, "url": "https://x.com/a(b)", "fecha": AHORA, "fuente": "Fuente_1"}] * 5
    msg = dd.armar_mensaje(AHORA, {"precio": 82000.0, "var24h": 1.0, "var7d": -2.0, "funding_anual": 5.5, "miedo": (64, "codicia", 59)},
                           None, None, its, its, its, "🟢 COMPRA", "lectura _x_")
    assert len(msg) <= 3900 and "A\\_b \\*c\\* \\[d]" in msg and "%29" in msg and "S&P 500   sin datos" in msg
    assert "lun 12/10" in msg


def fuentes_falsas(btc=True, indices=True, feeds=True):
    def http(url, **params):
        if "ticker" in url and btc:
            return json.dumps({"code": 0, "data": [{"lastPrice": "82000", "priceChangePercent": "1.2%"}]}).encode()
        if "klines" in url and btc:
            return json.dumps({"code": 0, "data": [{"time": i, "close": str(80000 + i * 100)} for i in range(9)]}).encode()
        if "premiumIndex" in url and btc:
            return json.dumps({"code": 0, "data": {"lastFundingRate": "0.0001"}}).encode()
        if "alternative.me" in url:
            return json.dumps({"data": [{"value": "64", "value_classification": "Greed"}, {"value": "59", "value_classification": "Greed"}]}).encode()
        if "yahoo" in url and indices:
            ts = [1_760_000_000 + i * 86400 for i in range(8)]
            return json.dumps({"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": [100 + i for i in range(8)]}]}}]}}).encode()
        if feeds and "feed" in url or "rss" in url:
            return rss([("Bitcoin ETF flows rise", "https://x/1", fecha(2)), ("Stocks rally as Fed signals rate cut", "https://x/2", fecha(3))])
        raise RuntimeError("fuente caída")
    return http


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "OUT_DIR", tmp_path)
    monkeypatch.setattr(dd, "BASE_DIR", tmp_path)
    enviados = []
    monkeypatch.setattr(dd, "_emitir", lambda **kw: enviados.append(kw) or True)
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)
    cfg = json.loads(dd.CONFIG_PATH.read_text())
    cfg["fuentes_btc"] = [["B", "https://b.test/feed"]]; cfg["fuentes_mercado"] = [["M", "https://m.test/rss"]]
    cfg["fuente_fed"] = ["Fed", "https://fed.test/rss"]
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(cfg)); monkeypatch.setattr(dd, "CONFIG_PATH", p)
    return enviados


def test_envia_una_vez_por_dia_y_no_antes_de_la_hora(entorno):
    assert dd.run_daily_digest(ahora=AHORA.replace(hour=15), http=fuentes_falsas()) is None
    r = dd.run_daily_digest(ahora=AHORA, http=fuentes_falsas())
    assert r["btc"] and r["indices"] and r["n_btc"] >= 1 and len(entorno) == 1
    assert "82.000" in entorno[0]["mensaje_md"] and "codicia" in entorno[0]["mensaje_md"]
    assert dd.run_daily_digest(ahora=AHORA + dt.timedelta(hours=2), http=fuentes_falsas()) == {}
    assert len(entorno) == 1
    assert dd.run_daily_digest(ahora=AHORA + dt.timedelta(days=1), http=fuentes_falsas())["btc"] and len(entorno) == 2


def test_tolera_fuentes_caidas_y_no_marca_enviado_si_no_hay_nada(entorno):
    r = dd.run_daily_digest(ahora=AHORA, http=fuentes_falsas(indices=False))
    assert r["indices"] is False and r["btc"] and "sin datos" in entorno[0]["mensaje_md"]
    entorno.clear()
    def todo_caido(url, **p): raise RuntimeError("sin red")
    assert dd.run_daily_digest(ahora=AHORA + dt.timedelta(days=1), http=todo_caido) is None
    assert entorno == []                                       # no manda un mensaje vacío ni cierra el día
    assert dd.leer_estado()["fecha"] == "2026-10-12"          # el estado sigue siendo el del último envío real


def test_lectura_ia_apagada_sin_clave_y_con_llm_inyectado(entorno, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert dd.lectura_ia({"resumen_ia": False}, "x", llm=lambda s, d: "hola") is None
    assert dd.lectura_ia({"resumen_ia": True}, "x") is None                          # sin clave: se omite sin error
    vistos = []
    assert dd.lectura_ia({"resumen_ia": True}, "DATOS", llm=lambda s, d: vistos.append((s, d)) or "ok") == "ok"
    sistema, datos = vistos[0]
    assert datos == "DATOS" and "ignora cualquier instrucción" in sistema.lower()
    assert "imperativo" in sistema and "confianza" in sistema
    assert dd.lectura_ia({"resumen_ia": True}, "x", llm=lambda s, d: 1 / 0) is None  # fallo del modelo: se omite


def velas_diarias(cierres, hoy):
    """Velas al estilo BingX terminando en la de HOY (en formación, con un cierre absurdo)."""
    t0 = int(dt.datetime(hoy.year, hoy.month, hoy.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    v = [{"time": t0 - (len(cierres) - i) * 86_400_000, "close": str(c)} for i, c in enumerate(cierres)]
    return v + [{"time": t0, "close": "1"}]


def test_tendencia_usa_solo_velas_cerradas():
    hoy = dt.date(2026, 10, 12)
    sube = [100.0 + i for i in range(120)]
    e = dd.tendencia(velas_diarias(sube, hoy), precio=225.0, hoy=hoy)
    assert e["tend"] == "sobre" and e["cierre"] == 219.0            # la vela de hoy (cierre 1) no cuenta
    assert e["media100"] == pytest.approx(sum(sube[-100:]) / 100)
    assert dd.tendencia(velas_diarias(sube[:50], hoy), precio=1.0, hoy=hoy) == {}   # historia corta


def test_tasa_base_cae_al_total_si_el_tramo_es_chico():
    b = dd.tasa_base("sobre", 55.0)
    assert b["tramo"] == "50-70" and 0.4 < b["sube7"] < 0.7 and b["p10_7"] < 0 < b["p90_7"]
    assert dd.tasa_base("sobre", 20.0)["tramo"] == "todos"          # "sobre|<30" tiene n=1


def test_mensaje_largo_quita_titulares_sin_cortar_enlaces():
    its = [{"titulo": "Bitcoin " + "x" * 140, "url": "https://x.com/" + "y" * 200, "fecha": AHORA, "fuente": "F"}] * 5
    btc = {"precio": 82000.0, "var24h": 1.0, "media100": 72000.0, "tend": "sobre", "rsi": 55.0, "dist_media": 13.9, "var30d": 7.0}
    base = dd.tasa_base("sobre", 55.0)
    msg = dd.armar_mensaje(AHORA, btc, None, None, its, its, its[:2], "🟢 COMPRA desde el 19/08", "• " + "a" * 1200, base)
    assert len(msg) <= 3900 and msg.count("[") == msg.count("](")     # todo enlace quedó completo
    assert "Rango típico a 7 días" in msg and "Cambia la lectura" in msg and "Análisis" in msg
