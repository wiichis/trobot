"""Gate horario de sesión.

Desactivado el 18/08/2026 vaciando `session.entry_hours_utc` en
`archivos/backtesting/configs/live_benchmark_runtime.json`. Bloqueaba 1 de cada 3
señales (35% y 33% medidos en dos semanas independientes), no filtraba horas malas
—el winrate de las bloqueadas era igual o mejor—, venía calibrado para otra
estrategia (`rsi_reversal` 30m_5m) y el backtest nunca lo simulaba: los params se
optimizan 24/7 y se ejecutaban 14/24.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from pkg.live_runtime_config import (
    get_allowed_entry_hours_utc,
    is_entry_hour_allowed_utc,
)

CONFIG = Path(__file__).resolve().parents[1] / 'archivos' / 'backtesting' / 'configs' / 'live_benchmark_runtime.json'
# Las que bloqueaba el perfil liquid_utc_wo_13_18_20.
HORAS_ANTES_BLOQUEADAS = (0, 1, 2, 3, 4, 5, 13, 18, 20, 23)


def test_la_config_deja_las_horas_vacias():
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    assert cfg['session']['entry_hours_utc'] == []


def test_no_hay_horas_permitidas_configuradas():
    """Lista vacía es la forma canónica de decir 24/7."""
    assert get_allowed_entry_hours_utc() == ()


def test_las_24_horas_estan_permitidas():
    for h in range(24):
        ts = datetime(2026, 8, 18, h, 30, tzinfo=timezone.utc)
        assert is_entry_hour_allowed_utc(ts), f'{h}h debería estar permitida'


def test_las_horas_que_antes_bloqueaba_ahora_pasan():
    """Las 5 señales perdidas la semana del 18/08 caían en estas horas."""
    for h in HORAS_ANTES_BLOQUEADAS:
        ts = datetime(2026, 8, 18, h, 30, tzinfo=timezone.utc)
        assert is_entry_hour_allowed_utc(ts), f'{h}h seguía bloqueada'


def test_naive_datetime_se_trata_como_utc():
    assert is_entry_hour_allowed_utc(datetime(2026, 8, 18, 3, 30)) is True


def test_sin_argumento_usa_ahora():
    assert is_entry_hour_allowed_utc() is True
