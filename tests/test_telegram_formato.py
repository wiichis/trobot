"""Formato de los avisos de Telegram (03/10/2026): Markdown con respaldo en texto limpio."""
import json
import re

import pandas as pd
import pytest

import pkg.btc_alertas as ba
import pkg.btc_bots as bb
import pkg.paper_momentum as pm
import pkg.telegram_alerts as ta
from pkg.lifecycle_events import _build_body, escapar_md


def _marcas_balanceadas(texto: str) -> bool:
    """Fuera de bloques ``` y de `código`, cada `*` y `_` sin escapar debe cerrar."""
    t = re.sub(r"```.*?```", "", texto, flags=re.S)
    t = re.sub(r"`[^`]*`", "", t)
    t = t.replace("\\_", "").replace("\\*", "")
    return t.count("*") % 2 == 0 and t.count("_") % 2 == 0 and "`" not in t


def test_cuerpo_generico_escapa_lo_que_romperia_el_markdown():
    body = _build_body({"reason": "sl_watch_stop_loss", "detail": "a*b `c` [d"})
    assert "sl\\_watch\\_stop\\_loss" in body and "a\\*b" in body
    assert _marcas_balanceadas(body)


def test_mensaje_armado_pasa_tal_cual_y_floats_sin_ceros_de_mas():
    assert _build_body({"mensaje_md": "*hola*", "otro": 1}) == "*hola*"
    assert "Restante  `0.03`" in _build_body({"remaining_qty": 0.03, "symbol": "BNB-USDT"}, "tp1_filled")


def test_quitar_markdown_deja_texto_legible():
    t = ta.quitar_markdown("✅ *TP1*\nPar `BNB`\nRazón: sl\\_watch\n_nota_\n```\nA 1\n```")
    assert t == "✅ TP1\nPar BNB\nRazón: sl_watch\nnota\n\nA 1\n"


class _Resp:
    def __init__(self, status):
        self.status_code, self.ok, self.text = status, status == 200, ""


def test_si_telegram_rechaza_el_formato_va_en_texto_limpio(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"telegram": {"enabled": True, "parse_mode": "Markdown",
                                            "state_file": str(tmp_path / "estado.json")}}))
    alerter = ta.TelegramAlerter(config_path=cfg, repo_root=tmp_path)
    alerter.token, alerter.chat_id = "t", "c"
    enviados = []
    def post(url, data, timeout):
        enviados.append(data)
        return _Resp(400 if "parse_mode" in data else 200)
    monkeypatch.setattr(ta.requests, "post", post)
    r = alerter.send(category="btc_ventana", severity="INFO", body="*roto", ts_utc="")
    assert r.sent and len(enviados) == 2
    assert "parse_mode" not in enviados[1] and "*" not in enviados[1]["text"]


def test_cabecera_con_emoji_propio_salvo_alarmas():
    a = ta.TelegramAlerter.__new__(ta.TelegramAlerter)
    assert a._build_header(severity="INFO", category="btc_ventana", ts_utc="") == "📈 *Ventana BTC*"
    assert a._build_header(severity="WARN", category="btc_ventana", ts_utc="").startswith("⚠️")


def _estado_ventana(**kw):
    e = {"barra": "2026-10-02", "cierre": 84480.0, "media100": 70761.0, "rsi": 62.9, "tendencia": 1,
         "tendencia_desde": "2026-08-19", "caida": 0, "caida_desde": "2026-01-01"}
    e.update(kw)
    return e


@pytest.mark.parametrize("prev,e", [
    (None, _estado_ventana()),
    (None, _estado_ventana(tendencia=0, caida=1, rsi=27.0)),
    (_estado_ventana(), _estado_ventana(tendencia=0, cierre=68900.0)),
    (_estado_ventana(), _estado_ventana(caida=1, rsi=28.0)),
    (_estado_ventana(caida=1), _estado_ventana(caida=0, rsi=50.5)),
])
def test_mensajes_de_ventana_bien_cerrados(prev, e):
    assert _marcas_balanceadas(ba.mensaje_md(prev, e))
    assert _marcas_balanceadas(ba.linea_md(e))


def test_mensajes_de_bots_bien_cerrados_y_sin_menos_cero():
    cfg = json.loads(bb.CONFIG_PATH.read_text(encoding="utf-8"))
    estado = {"bots": {k: bb.libro_nuevo(46.5) for k in cfg["bots"]}}
    ops = [{"bot": "A", "nombre": "tendencia lenta", "accion": "da vuelta: cierra SHORT, abre LONG",
            "resultado_cerrada": -0.41},
           {"bot": "B", "nombre": "tendencia rápida", "accion": "CORTE: perdió 30% o más; cierra y se apaga",
            "resultado_cerrada": -14.0}]
    md = bb.mensaje_operaciones_md(cfg, estado, ops, 84592.4, -0.00001, "papel")
    assert _marcas_balanceadas(md) and "-0,0" not in md and "🔄" in md and "🛑" in md
    md = bb.mensaje_resumen_md(cfg, estado, 84592.4, 0.0007, "papel", ba.linea_md(_estado_ventana()))
    assert _marcas_balanceadas(md) and "📈 Ventana BTC" in md
    assert _marcas_balanceadas(bb.mensaje_resumen_md(cfg, estado, 84592.4, 0.0, "papel", None, inicio=True))


def test_mensaje_de_momentum_bien_cerrado():
    md = pm.mensaje_md({"rebalanceo": 2, "largos": ["PUMP-USDT", "A_B-USDT"], "cortos": ["XPL-USDT"],
                        "proximo": "2026-10-08 23:00:00+00:00"}, 0.0035, 0.35)
    assert _marcas_balanceadas(md) and "08/10 23:00 UTC" in md
