"""Banca de pares (01/10/2026): criterios y velas, aisladas del camino de órdenes."""
import json
from datetime import date

import pandas as pd
import pytest

import pkg.bench as bench
import pkg.price_bingx_5m as px


# ---------------------------------------------------------------- criterios


def test_avax_del_01_10_va_a_banca():
    """Los números reales del 01/10: sim −8,99/−10,26/+2,26/−0,49, real 4/4 bloques negativos."""
    ok, motivo = bench.debe_ir_a_banca({'30d': -8.99, '60d': -10.26, '90d': 2.26, '120d': -0.49},
                                       [-2.93, -0.07, -0.38, -0.86])
    assert ok, motivo


def test_bnb_del_01_10_no_va_a_banca_aunque_pierda_en_real():
    """Real negativo 4/4 pero el sim lo da positivo en 3/4: no alcanza."""
    ok, _ = bench.debe_ir_a_banca({'30d': 3.14, '60d': 3.50, '90d': 3.60, '120d': -5.37},
                                  [-0.50, -1.58, -1.54, -0.66])
    assert not ok


def test_un_bloque_sin_operar_no_cuenta_como_negativo():
    ok, _ = bench.debe_ir_a_banca({'30d': -1, '60d': -1, '90d': -1, '120d': -1},
                                  [0.0, 0.0, -0.5, -0.5])
    assert not ok


def test_sin_las_cuatro_ventanas_no_decide():
    ok, _ = bench.debe_ir_a_banca({'30d': -1, '60d': -1, '90d': -1}, [-1, -1, -1, -1])
    assert not ok


def test_usa_los_ultimos_cuatro_bloques():
    ok, _ = bench.debe_ir_a_banca({'30d': -1, '60d': -1, '90d': -1, '120d': -1},
                                  [-1, -1, -1, 1.0, 0.5, -1, -1])
    assert not ok


def test_activacion_exige_protocolo():
    ok, motivo = bench.puede_activarse(False, 40, 0.01)
    assert not ok and 'protocolo' in motivo


def test_activacion_exige_muestra_forward():
    ok, motivo = bench.puede_activarse(True, 14, 0.01)
    assert not ok and 'faltan 1' in motivo


def test_activacion_exige_superar_el_sesgo_del_sim():
    assert not bench.puede_activarse(True, 20, bench.SESGO_SIM_POR_TRADE_PCT)[0]
    assert bench.puede_activarse(True, 20, bench.SESGO_SIM_POR_TRADE_PCT + 0.0001)[0]


def test_vencimiento_en_banca():
    assert not bench.vencio_en_banca('2026-10-02', hoy=date(2026, 12, 1))
    assert bench.vencio_en_banca('2026-10-02', hoy=date(2026, 12, 2))


# ---------------------------------------------------------------- lectura


def _escribir(tmp_path, rows):
    p = tmp_path / 'bench.json'
    p.write_text(json.dumps(rows), encoding='utf-8')
    return p


def test_banca_inexistente_es_vacia(tmp_path):
    assert bench.load_bench(tmp_path / 'no.json') == []


def test_estado_desconocido_rompe(tmp_path):
    p = _escribir(tmp_path, [{'symbol': 'avax-usdt', 'estado': 'activo'}])
    with pytest.raises(ValueError):
        bench.load_bench(p)


def test_symbols_normaliza_y_excluye_activos(tmp_path):
    p = _escribir(tmp_path, [{'symbol': 'avax-usdt', 'estado': 'banca'},
                             {'symbol': 'UNI-USDT', 'estado': 'observacion'}])
    assert bench.bench_symbols(p) == ['AVAX-USDT', 'UNI-USDT']
    assert bench.bench_symbols(p, excluir=['UNI-USDT']) == ['AVAX-USDT']


# ---------------------------------------------------------------- velas de la banca


AHORA = pd.Timestamp('2026-10-02 12:00', tz='UTC')


def _velas(symbol, fin, n):
    fechas = [fin - pd.Timedelta(minutes=5 * (n - 1 - i)) for i in range(n)]
    return [dict(symbol=symbol, open=1.0, high=1.1, low=0.9, close=1.0, volume=10.0, date=f)
            for f in fechas]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    bp = _escribir(tmp_path, [{'symbol': 'AVAX-USDT', 'estado': 'banca'},
                              {'symbol': 'ETH-USDT', 'estado': 'observacion'}])
    monkeypatch.setattr(bench, 'BENCH_PATH', bp)
    monkeypatch.setattr(px, 'BENCH_CSV_PATH', tmp_path / 'bench.csv')
    monkeypatch.setattr(px, 'CSV_PATH', tmp_path / 'trading.csv')
    monkeypatch.setattr(px, 'currencies_list', lambda: ['ETH-USDT', 'BCH-USDT'])
    monkeypatch.setattr(px.time, 'sleep', lambda *_: None)
    pedidos = []

    def fake_fetch(symbol, limit, end_time_ms=None):
        pedidos.append((symbol, limit, end_time_ms))
        fin = AHORA if end_time_ms is None else pd.Timestamp(end_time_ms, unit='ms', tz='UTC')
        return _velas(symbol, fin, limit)

    monkeypatch.setattr(px, '_fetch_bingx_candles', fake_fetch)
    return tmp_path, pedidos


def test_sync_baja_solo_pares_en_banca_y_no_toca_el_csv_de_trading(entorno):
    tmp, pedidos = entorno
    px.sync_bench_candles(now_utc=AHORA)
    df = pd.read_csv(tmp / 'bench.csv')
    assert set(df.symbol) == {'AVAX-USDT'}           # ETH opera: no se duplica
    assert {p[0] for p in pedidos} == {'AVAX-USDT'}
    assert not (tmp / 'trading.csv').exists()


def test_sync_rellena_hacia_atras_con_presupuesto(entorno, monkeypatch):
    tmp, pedidos = entorno
    monkeypatch.setattr(px, 'BENCH_MAX_REQUESTS_PER_RUN', 4)
    px.sync_bench_candles(now_utc=AHORA)
    assert len(pedidos) == 4                         # 1 reciente + 3 de backfill, y corta
    df = pd.read_csv(tmp / 'bench.csv')
    assert len(df) == 36 + 3 * 1000                   # tramos contiguos, sin solaparse


def test_sync_es_idempotente_y_reemplaza_la_vela_parcial(entorno, monkeypatch):
    tmp, _ = entorno
    monkeypatch.setattr(px, 'BENCH_MAX_REQUESTS_PER_RUN', 1)
    px.sync_bench_candles(now_utc=AHORA)
    n1 = len(pd.read_csv(tmp / 'bench.csv'))
    assert px.sync_bench_candles(now_utc=AHORA) == 0
    assert len(pd.read_csv(tmp / 'bench.csv')) == n1


def test_sync_nunca_levanta_excepcion(entorno, monkeypatch):
    def roto(*a, **k):
        raise RuntimeError('API caída')
    monkeypatch.setattr(px, '_fetch_bingx_candles', roto)
    assert px.sync_bench_candles(now_utc=AHORA) == 0


def test_sync_con_bench_corrupto_no_tira_el_bot(entorno, monkeypatch):
    tmp, _ = entorno
    (tmp / 'bench.json').write_text('{roto', encoding='utf-8')
    assert px.sync_bench_candles(now_utc=AHORA) == -1


def test_la_purga_del_long_preserva_la_banca(entorno, monkeypatch):
    monkeypatch.setattr(px, 'MAX_BACKFILL_BATCHES', 0)
    viejo = AHORA - pd.Timedelta(days=60)
    df = pd.DataFrame([dict(symbol=s, open=1, high=1, low=1, close=1, volume=1, date=viejo)
                       for s in ('BCH-USDT', 'AVAX-USDT', 'DOT-USDT')])
    out = px._ensure_long_history(df, AHORA)
    assert set(out.symbol) == {'BCH-USDT', 'AVAX-USDT'}   # DOT retirado sí se depura


def test_sync_prioriza_lo_reciente_de_todos_sobre_el_relleno(tmp_path, monkeypatch):
    """Caso real del 02/10: el relleno de los primeros pares agotó el presupuesto."""
    syms = [f'P{i}-USDT' for i in range(6)]
    bp = _escribir(tmp_path, [{'symbol': s, 'estado': 'banca'} for s in syms])
    monkeypatch.setattr(bench, 'BENCH_PATH', bp)
    monkeypatch.setattr(px, 'BENCH_CSV_PATH', tmp_path / 'bench.csv')
    monkeypatch.setattr(px, 'currencies_list', lambda: [])
    monkeypatch.setattr(px.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(px, 'BENCH_MAX_REQUESTS_PER_RUN', 8)

    def fake_fetch(symbol, limit, end_time_ms=None):
        fin = AHORA if end_time_ms is None else pd.Timestamp(end_time_ms, unit='ms', tz='UTC')
        return _velas(symbol, fin, limit)

    monkeypatch.setattr(px, '_fetch_bingx_candles', fake_fetch)
    px.sync_bench_candles(now_utc=AHORA)
    df = pd.read_csv(tmp_path / 'bench.csv')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    ultimas = df.groupby('symbol')['date'].max()
    assert set(ultimas.index) == set(syms)            # todos tienen lo reciente
    assert (ultimas == AHORA).all()


def test_sync_completa_la_historia_de_un_par_que_ya_tiene_velas_recientes(tmp_path, monkeypatch):
    """Caso real del 02/10: con velas recientes guardadas, el relleno nunca miraba hacia atrás."""
    bp = _escribir(tmp_path, [{'symbol': 'ENA-USDT', 'estado': 'observacion'}])
    monkeypatch.setattr(bench, 'BENCH_PATH', bp)
    monkeypatch.setattr(px, 'BENCH_CSV_PATH', tmp_path / 'bench.csv')
    monkeypatch.setattr(px, 'currencies_list', lambda: [])
    monkeypatch.setattr(px.time, 'sleep', lambda *_: None)
    pd.DataFrame(_velas('ENA-USDT', AHORA - pd.Timedelta(minutes=5), 200)).to_csv(tmp_path / 'bench.csv', index=False)
    pedidos = []

    def fake_fetch(symbol, limit, end_time_ms=None):
        pedidos.append(end_time_ms)
        fin = AHORA if end_time_ms is None else pd.Timestamp(end_time_ms, unit='ms', tz='UTC')
        return _velas(symbol, fin, limit)

    monkeypatch.setattr(px, '_fetch_bingx_candles', fake_fetch)
    px.sync_bench_candles(now_utc=AHORA)
    df = pd.read_csv(tmp_path / 'bench.csv')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    assert df['date'].min() <= AHORA - pd.Timedelta(days=px.BENCH_BACKFILL_DAYS)
    assert df['date'].max() == AHORA
    # contiguo: ninguna vela faltante entre la más vieja y la más nueva
    assert (df['date'].diff().dropna() == pd.Timedelta(minutes=5)).all()


def test_sync_ya_completo_no_gasta_pedidos_de_relleno(tmp_path, monkeypatch):
    bp = _escribir(tmp_path, [{'symbol': 'ENA-USDT', 'estado': 'observacion'}])
    monkeypatch.setattr(bench, 'BENCH_PATH', bp)
    monkeypatch.setattr(px, 'BENCH_CSV_PATH', tmp_path / 'bench.csv')
    monkeypatch.setattr(px, 'currencies_list', lambda: [])
    n = px.BENCH_BACKFILL_DAYS * 288 + 10
    pd.DataFrame(_velas('ENA-USDT', AHORA - pd.Timedelta(minutes=5), n)).to_csv(tmp_path / 'bench.csv', index=False)
    pedidos = []

    def fake_fetch(symbol, limit, end_time_ms=None):
        pedidos.append(end_time_ms)
        return _velas(symbol, AHORA, limit)

    monkeypatch.setattr(px, '_fetch_bingx_candles', fake_fetch)
    px.sync_bench_candles(now_utc=AHORA)
    assert pedidos == [None]                         # sólo lo reciente


def test_habilitar_en_indicadores_destraba_las_senales_de_un_par_fuera_de_best_prod(monkeypatch):
    """Sin habilitarlo, _calc_symbol apaga las señales de todo par que no opera."""
    import pkg.indicadores as ind
    monkeypatch.setattr(ind, 'TRADE_SYMBOLS', ['BCH-USDT'])
    bench.habilitar_en_indicadores(['uni-usdt'])
    assert 'UNI-USDT' in ind.TRADE_SYMBOLS and 'BCH-USDT' in ind.TRADE_SYMBOLS
