"""Reparto de tramos de TP en el parity-sim.

Contexto (31/08/2026): `_prepare_live_tp_plan` usaba `TP_SPLITS_DEFAULT` (40/40/20),
una constante del módulo, mientras el live reparte según `tp_partial_distribution` del
runtime config (hoy 33/33/34). Dos consecuencias:

- El parity simulaba un ladder distinto al que corre en prod.
- El reparto quedaba fuera del alcance de `TROBOT_RUNTIME_CONFIG_PATH`, así que
  cualquier A/B sobre la distribución era un **no-op silencioso** — se corría, daba
  resultados idénticos y parecía que el cambio "no tenía efecto".
"""
import json

import pytest

from pkg.backtesting import TP_SPLITS_DEFAULT, _runtime_tp_splits_parity


def test_toma_el_reparto_del_runtime_no_la_constante():
    """El live usa 33/33/34; la constante del módulo es 40/40/20."""
    splits = _runtime_tp_splits_parity()
    assert len(splits) == 3
    assert splits == pytest.approx((0.33, 0.33, 0.34), abs=1e-6)
    assert splits != pytest.approx(TP_SPLITS_DEFAULT, abs=1e-6)


def test_el_reparto_suma_uno():
    assert sum(_runtime_tp_splits_parity()) == pytest.approx(1.0)


def test_coincide_con_lo_que_reparte_el_live():
    """Misma fuente y misma normalización que `_runtime_tp_splits` de monkey_bx."""
    from pkg.monkey_bx import _runtime_tp_splits
    assert _runtime_tp_splits_parity() == pytest.approx(_runtime_tp_splits(), abs=1e-9)


def test_el_override_de_runtime_config_ahora_tiene_efecto(tmp_path, monkeypatch):
    """Un A/B sobre la distribución debe cambiar el reparto, no ser un no-op."""
    import pkg.live_runtime_config as lrc

    cfg = tmp_path / 'runtime.json'
    cfg.write_text(json.dumps({'execution_tp': {'tp_partial_distribution': [0.6, 0.3, 0.1]}}),
                   encoding='utf-8')
    monkeypatch.setattr(lrc, 'DEFAULT_CONFIG_PATH', cfg)
    lrc.reload_live_runtime_config()

    assert _runtime_tp_splits_parity() == pytest.approx((0.6, 0.3, 0.1), abs=1e-6)


def test_normaliza_una_distribucion_que_no_suma_uno(tmp_path, monkeypatch):
    import pkg.live_runtime_config as lrc

    cfg = tmp_path / 'runtime.json'
    cfg.write_text(json.dumps({'execution_tp': {'tp_partial_distribution': [3, 3, 4]}}),
                   encoding='utf-8')
    monkeypatch.setattr(lrc, 'DEFAULT_CONFIG_PATH', cfg)
    lrc.reload_live_runtime_config()

    assert _runtime_tp_splits_parity() == pytest.approx((0.3, 0.3, 0.4), abs=1e-6)


@pytest.mark.parametrize('malo', [[], [0.5, 0.5], [0, 0, 0]])
def test_config_invalida_no_rompe_y_da_un_reparto_usable(tmp_path, monkeypatch, malo):
    """Una distribución inválida nunca debe propagarse ni reventar.

    `live_runtime_config` ya la sanea antes de que el parity la vea, así que el fallback
    efectivo es el default del runtime (33/33/34), no la constante legacy del módulo.
    Lo que importa es el contrato: 3 tramos positivos que suman 1.
    """
    import pkg.live_runtime_config as lrc

    cfg = tmp_path / 'runtime.json'
    cfg.write_text(json.dumps({'execution_tp': {'tp_partial_distribution': malo}}),
                   encoding='utf-8')
    monkeypatch.setattr(lrc, 'DEFAULT_CONFIG_PATH', cfg)
    lrc.reload_live_runtime_config()

    splits = _runtime_tp_splits_parity()
    assert len(splits) == 3
    assert all(v > 0 for v in splits)
    assert sum(splits) == pytest.approx(1.0)


def test_la_constante_legacy_sigue_siendo_el_fallback_ultimo(monkeypatch):
    """Si el runtime config no se puede leer, se cae a la constante del módulo."""
    def _explota():
        raise RuntimeError('config ilegible')

    monkeypatch.setattr('pkg.live_runtime_config.get_tp_partial_distribution', _explota)
    assert _runtime_tp_splits_parity() == pytest.approx(TP_SPLITS_DEFAULT, abs=1e-9)
