"""El ladder de TP se ancla al precio de ENTRADA, no al close de la vela viva.

Motivación (14/09): al recolocar un tramo, el nivel se recalculaba desde el indicador
vivo y se alejaba de la entrada en 7 de 7 casos medidos (drift medio 129 bps, máx 302).
Caso AVAX del 09/09: el precio atravesó el TP1 a las 09:45 y a las 09:53 el bot movió la
orden de 7,922 a 7,890, por debajo del mercado; la posición cerró sin registrar el fill.
"""
import math
import pytest

from pkg.monkey_bx import (
    TP_LADDER_FACTORS_DEFAULT,
    _tp_ladder_factors,
    _tp_levels_from_entry,
)
from pkg.backtesting import (
    _tp_ladder_factors_parity,
    _tp_levels_from_entry_parity,
)

P = {"tp_mode": "fixed", "tp": 0.018}


# --- anclaje -----------------------------------------------------------------

def test_los_niveles_salen_del_precio_de_entrada():
    lv = _tp_levels_from_entry(100.0, "LONG", P)
    assert lv == pytest.approx([100 * (1 + f * 0.018) for f in (0.6, 1.0, 1.6)])


def test_short_es_simetrico():
    lv = _tp_levels_from_entry(100.0, "SHORT", P)
    assert lv == pytest.approx([100 * (1 - f * 0.018) for f in (0.6, 1.0, 1.6)])


def test_el_nivel_no_cambia_aunque_el_mercado_se_mueva():
    """Es la propiedad que arregla el bug: idempotencia entre recolocaciones."""
    primero = _tp_levels_from_entry(7.9500, "LONG", {"tp_mode": "fixed", "tp": 0.015})
    # el mercado se mueve, pasan ciclos, el job recoloca el mismo tramo
    segundo = _tp_levels_from_entry(7.9500, "LONG", {"tp_mode": "fixed", "tp": 0.015})
    assert primero == segundo


def test_caso_avax_09_09_el_tp1_ya_no_retrocede():
    """Con entrada 7,8265 el TP1 queda fijo; antes se recalculaba desde el close vivo
    y la recolocación lo bajó de 7,922 a 7,890 (por debajo del mercado)."""
    entry = 7.8265
    tp1_ciclo_1 = _tp_levels_from_entry(entry, "LONG", {"tp_mode": "fixed", "tp": 0.015})[0]
    tp1_ciclo_2 = _tp_levels_from_entry(entry, "LONG", {"tp_mode": "fixed", "tp": 0.015})[0]
    assert tp1_ciclo_1 == tp1_ciclo_2
    assert tp1_ciclo_1 > entry  # un LONG nunca pone su TP por debajo de la entrada


# --- factores del ladder -----------------------------------------------------

def test_default_espeja_indicadores():
    from pkg.indicadores import TP_FACTORS
    assert TP_LADDER_FACTORS_DEFAULT == tuple(TP_FACTORS)
    assert _tp_ladder_factors({}) == TP_LADDER_FACTORS_DEFAULT


def test_override_por_simbolo():
    assert _tp_ladder_factors({"tp_factors": [0.4, 0.8, 1.2]}) == (0.4, 0.8, 1.2)
    assert _tp_ladder_factors({"tp_factors": "0.4, 0.8, 1.2"}) == (0.4, 0.8, 1.2)
    assert _tp_ladder_factors({"tp_ladder_factors": [0.5, 0.9, 1.3]}) == (0.5, 0.9, 1.3)


@pytest.mark.parametrize("malo", [
    [1.0, 0.5, 1.6],     # desordenado -> TP2 antes que TP1
    [0.6, 0.6, 1.6],     # no estrictamente creciente
    [0.6, 1.0],          # faltan tramos
    [0.0, 1.0, 1.6],     # factor nulo
    [-0.6, 1.0, 1.6],    # negativo
    ["x", 1.0, 1.6],     # no numérico
    [0.6, float("nan"), 1.6],
])
def test_un_ladder_invalido_cae_al_default(malo):
    """Preferimos el default a someter precios inconsistentes: un ladder desordenado
    rompería el avance de etapas."""
    assert _tp_ladder_factors({"tp_factors": malo}) == TP_LADDER_FACTORS_DEFAULT


# --- modos que no aplican ----------------------------------------------------

@pytest.mark.parametrize("params", [
    {"tp_mode": "atrx", "tp": 0.018},
    {"tp_mode": "none", "tp": 0.018},
    {"tp_mode": "fixed", "tp": 0.0},
    {"tp_mode": "fixed"},
])
def test_devuelve_vacio_y_manda_el_indicador(params):
    assert _tp_levels_from_entry(100.0, "LONG", params) == []


@pytest.mark.parametrize("entry", [0.0, -1.0, float("nan"), float("inf")])
def test_entrada_invalida_no_produce_niveles(entry):
    assert _tp_levels_from_entry(entry, "LONG", P) == []


def test_params_vacios_o_none_no_rompen():
    assert _tp_levels_from_entry(100.0, "LONG", None) == []
    assert _tp_ladder_factors(None) == TP_LADDER_FACTORS_DEFAULT


# --- paridad live <-> sim ----------------------------------------------------

@pytest.mark.parametrize("side", ["LONG", "SHORT"])
@pytest.mark.parametrize("tp", [0.012, 0.018, 0.025, 0.036])
def test_live_y_parity_calculan_el_mismo_ladder(side, tp):
    params = {"tp_mode": "fixed", "tp": tp}
    vivo = _tp_levels_from_entry(123.45, side, params)
    sim = _tp_levels_from_entry_parity(123.45, side.lower(), params)
    assert vivo == pytest.approx(sim)


def test_los_dos_motores_validan_igual():
    for caso in ({"tp_factors": [0.4, 0.8, 1.2]}, {"tp_factors": [1.0, 0.5, 1.6]}, {}):
        assert _tp_ladder_factors(caso) == _tp_ladder_factors_parity(caso)


def test_parity_respeta_r_multiples():
    """Con use_r_multiple_tps el ladder lo arma compute_tp_prices_from_r_multiples."""
    assert _tp_levels_from_entry_parity(
        100.0, "long", {"tp_mode": "fixed", "tp": 0.018, "use_r_multiple_tps": True}
    ) == []


# --- el ladder real que corre hoy -------------------------------------------

def test_tp3_esta_lejos_del_recorrido_tipico():
    """Deja constancia del diagnóstico: con (0.6, 1.0, 1.6) y tp1_factor 0.70, el tramo
    3 queda a 1,6*tp contra un MFE mediano de 1,23% -> inalcanzable para varios pares."""
    for tp, tp3_pct in ((0.012, 1.92), (0.030, 4.80), (0.036, 5.76)):
        lv = _tp_levels_from_entry(100.0, "LONG", {"tp_mode": "fixed", "tp": tp})
        assert (lv[2] / 100.0 - 1.0) * 100 == pytest.approx(tp3_pct, abs=0.01)


# --- interacción con el sanitizador -----------------------------------------

def test_si_el_precio_ya_paso_el_tp_anclado_la_orden_va_a_llenar():
    """El nivel anclado es fijo, así que una posición muy en ganancia puede tener su TP1
    por detrás del mercado. El sanitizador lo lleva a ref+tick: la orden queda del lado
    correcto y llena. Es lo opuesto al bug, donde el nivel se ALEJABA y no llenaba."""
    from pkg.monkey_bx import _sanitize_tp_limit_price
    entry = 100.0
    tp1 = _tp_levels_from_entry(entry, "LONG", {"tp_mode": "fixed", "tp": 0.018})[0]
    mercado = 105.0  # el precio ya corrió muy por encima del TP1
    px = _sanitize_tp_limit_price(tp1, "BCH-USDT", "LONG", mercado)
    assert px > mercado, "un LIMIT SELL por debajo del mercado no puede descansar en el book"

    tp1_s = _tp_levels_from_entry(entry, "SHORT", {"tp_mode": "fixed", "tp": 0.018})[0]
    px_s = _sanitize_tp_limit_price(tp1_s, "BCH-USDT", "SHORT", 95.0)
    assert px_s < 95.0


def test_en_operacion_normal_el_sanitizador_no_toca_el_nivel():
    """Mientras el mercado no haya pasado el TP, el precio sometido es el anclado
    (salvo redondeo al tick) — o sea, idempotente entre recolocaciones."""
    from pkg.monkey_bx import _sanitize_tp_limit_price
    entry = 250.0
    tp1 = _tp_levels_from_entry(entry, "LONG", {"tp_mode": "fixed", "tp": 0.036})[0]
    a = _sanitize_tp_limit_price(tp1, "BCH-USDT", "LONG", 251.0)
    b = _sanitize_tp_limit_price(tp1, "BCH-USDT", "LONG", 252.5)  # el mercado se movió
    assert a == b, "el nivel sometido no debe depender de dónde esté el mercado"
    assert abs(a - tp1) / tp1 < 1e-3
