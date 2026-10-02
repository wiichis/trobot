"""Optimizador sobre el parity (02/10/2026): las piezas que deciden, sin simular.

Lo que se protege: que la elección dependa SÓLO del entrenamiento, que el test y las
falsaciones no se pisen con él, y que el veredicto compare contra "no tocar nada".
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "optimizar_parity", Path(__file__).resolve().parent.parent / "scripts" / "optimizar_parity.py")
op = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(op)

FIN = pd.Timestamp("2026-10-01 14:55", tz="UTC")


def test_las_ventanas_no_se_pisan():
    W = op.ventanas(FIN)
    assert W["test"][1] == FIN
    assert W["train"][1] == W["test"][0]                       # el train termina donde empieza el test
    assert (W["test"][1] - W["test"][0]).days == op.TEST_DIAS
    assert (W["train"][1] - W["train"][0]).days == op.TRAIN_DIAS
    for h in ("hold1", "hold2"):
        assert W[h][1] <= W["train"][0]                        # las falsaciones son anteriores al train


def test_tercios_cubren_el_entrenamiento():
    a, b = op.ventanas(FIN)["train"]
    t = op.tercios(a, b)
    assert t[0][0] == a and t[-1][1] == b and t[0][1] == t[1][0]


@pytest.mark.parametrize("res, esperado", [
    ({"pnl": 5.0, "n": 20, "tercios": [1, 2, 2]}, 5.0),
    ({"pnl": 5.0, "n": 20, "tercios": [-1, 1, 5]}, 5.0),       # 2 de 3 alcanza
    ({"pnl": 5.0, "n": 20, "tercios": [-1, -1, 7]}, None),     # un solo tercio: no
    ({"pnl": 5.0, "n": 9, "tercios": [1, 2, 2]}, None),        # pocos trades
])
def test_elegibilidad(res, esperado):
    assert op.puntaje_train(res) == esperado


def test_ranking_usa_solo_el_entrenamiento():
    trials = [
        {"id": "a", "train": {"pnl": 3.0, "n": 20, "tercios": [1, 1, 1]}, "test": {"pnl": 99}},
        {"id": "b", "train": {"pnl": 5.0, "n": 20, "tercios": [1, 1, 3]}, "test": {"pnl": -99}},
        {"id": "c", "train": {"pnl": 9.0, "n": 5, "tercios": [3, 3, 3]}},  # n < mínimo
    ]
    assert [t["id"] for t in op.ranking(trials)] == ["b", "a"]


def _r(test, h1, h2, n=10):
    return {"test": {"pnl": test, "n": n}, "hold1": {"pnl": h1}, "hold2": {"pnl": h2}}


def test_veredicto_sin_vigente():
    assert op.veredicto(_r(1, 1, 1), None)[0]
    assert not op.veredicto(_r(1, -1, 1), None)[0]
    assert not op.veredicto(_r(1, 1, 1, n=op.MIN_TRADES_TEST - 1), None)[0]


def test_veredicto_exige_ganarle_al_vigente():
    ok, checks = op.veredicto(_r(2, 2, 2), _r(3, 1, 1))
    assert not ok and not checks["le_gana_al_vigente_en_test"]
    assert op.veredicto(_r(4, 2, 2), _r(3, 1, 1))[0]


def test_muestreo_reproducible_y_sin_emas_cruzadas():
    esp = {"ema_fast": [8, 21, 50], "ema_slow": [30, 50], "tp": [0.01, 0.02]}
    a, b = op.muestrear(esp, 6, 7), op.muestrear(esp, 6, 7)
    assert a == b
    assert all(p["ema_fast"] < p["ema_slow"] for p in a)
    assert len({tuple(sorted(p.items())) for p in a}) == len(a)


def test_una_ventana_sin_datos_no_cuenta_como_aprobada():
    cand = _r(1, 1, 1)
    cand["hold2"] = {"pnl": None, "n": 0, "sin_datos": True}
    ok, checks = op.veredicto(cand, None)
    assert not ok and not checks["hold2_positivo"]
