"""Override de peso por par, con el piso que impone el escalonado de TP.

El tramo más chico del ladder tiene que superar el notional mínimo de cierre de BingX
(~7 USDT). Un peso por debajo de eso no rompe nada visible: simplemente deja de
escalonar. Por eso el override se acota y se avisa, en vez de aceptarse en silencio.
"""
import pytest
from pkg.monkey_bx import _peso_para_simbolo, _peso_minimo_por_escalonado

EQUAL = 0.20
BAL = 188.12


def _pbs(**kw):
    return {k.upper(): v for k, v in kw.items()}


def test_sin_override_devuelve_el_equal_weight():
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, {}, BAL) == EQUAL
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, _pbs(**{"ondo-usdt": {}}), BAL) == EQUAL


def test_override_valido_se_respeta():
    p = _pbs(**{"ondo-usdt": {"peso": 0.13}})
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, p, BAL) == pytest.approx(0.13)


def test_alias_weight():
    p = _pbs(**{"ondo-usdt": {"weight": 0.15}})
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, p, BAL) == pytest.approx(0.15)


def test_solo_afecta_al_par_que_lo_declara():
    p = _pbs(**{"ondo-usdt": {"peso": 0.13}})
    assert _peso_para_simbolo("BCH-USDT", EQUAL, p, BAL) == EQUAL


def test_el_piso_depende_del_balance():
    """Es la razón de calcularlo en vivo: si la cuenta baja, el piso sube."""
    alto = _peso_minimo_por_escalonado(100.0)
    bajo = _peso_minimo_por_escalonado(400.0)
    assert alto > bajo
    assert _peso_minimo_por_escalonado(BAL) == pytest.approx(0.1128, abs=0.002)


def test_un_peso_bajo_el_piso_se_sube_al_piso():
    p = _pbs(**{"ondo-usdt": {"peso": 0.05}})
    got = _peso_para_simbolo("ONDO-USDT", EQUAL, p, BAL)
    assert got == pytest.approx(_peso_minimo_por_escalonado(BAL))
    assert got > 0.05


def test_el_recorte_emite_telemetria(monkeypatch):
    import pkg.monkey_bx as m
    vistos = []
    monkeypatch.setattr(m, "emit_lifecycle_event", lambda *a, **k: vistos.append((a, k)))
    _peso_para_simbolo("ONDO-USDT", EQUAL, _pbs(**{"ondo-usdt": {"peso": 0.02}}), BAL)
    assert vistos, "un recorte silencioso es justo el fallo que se quiere evitar"
    assert vistos[0][0][0] == "peso_override_acotado"


def test_no_emite_cuando_no_hace_falta(monkeypatch):
    import pkg.monkey_bx as m
    vistos = []
    monkeypatch.setattr(m, "emit_lifecycle_event", lambda *a, **k: vistos.append(a))
    _peso_para_simbolo("ONDO-USDT", EQUAL, _pbs(**{"ondo-usdt": {"peso": 0.13}}), BAL)
    assert not vistos


def test_se_respeta_el_cap_individual():
    p = _pbs(**{"ondo-usdt": {"peso": 0.90}})
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, p, BAL, max_per_trade=0.50) == pytest.approx(0.50)


@pytest.mark.parametrize("malo", [0, -0.1, "x", None, float("nan"), float("inf")])
def test_override_invalido_cae_al_equal_weight(malo):
    p = _pbs(**{"ondo-usdt": {"peso": malo}})
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, p, BAL) == EQUAL


@pytest.mark.parametrize("bal", [0, -5, float("nan")])
def test_balance_invalido_no_rompe(bal):
    assert _peso_minimo_por_escalonado(bal) == 0.0
    p = _pbs(**{"ondo-usdt": {"peso": 0.13}})
    assert _peso_para_simbolo("ONDO-USDT", EQUAL, p, bal) == pytest.approx(0.13)


def test_el_peso_de_ondo_en_produccion_deja_el_tramo_sobre_el_minimo():
    """Guardia sobre el valor que se desplegó: si alguien lo baja más, este test avisa."""
    import json, os
    from pkg.live_runtime_config import get_tp_min_close_notional_usdt
    from pkg.monkey_bx import _runtime_tp_splits
    ruta = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pkg", "best_prod.json")
    params = {e["symbol"]: e["params"] for e in json.load(open(ruta))}
    peso = params["ONDO-USDT"].get("peso")
    assert peso is not None, "ONDO debería llevar override de peso"
    tramo = BAL * peso * min(x for x in _runtime_tp_splits() if x > 0)
    assert tramo >= get_tp_min_close_notional_usdt()
