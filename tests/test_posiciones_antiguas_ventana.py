"""`filtrando_posiciones_antiguas` — la ventana de tiempo del job de BE/ratchet.

Contexto (06/09/2026): el filtro usaba `pd.Timestamp.now() - timedelta(hours=9)` como
"ahora", un parche de zona horaria (había un `+5` comentado para Mac). En un server UTC
no corrige nada y abre una **ventana ciega de 9 horas**: una posición sólo entraba al job
cuando su STOP_MARKET tenía más de 9 h de antigüedad.

Y como el BE y el ratchet **recrean** la orden de stop al moverla, cada ajuste reiniciaba
el bloqueo. Efecto medido: el BE se armaba en el 16% de las posiciones — la mayoría
cierra antes de las 9 h y el job nunca las miraba.

Caso real que lo destapó (BNB-USDT SHORT):
  05/09 20:39  entra, se crea el SL
  06/09 05:43  recién ahí pasa el filtro (9 h después) -> BE dispara
  06/09 05:54  el BE crea un SL nuevo -> reinicia el bloqueo hasta las 14:54
  06/09 05:59  TP1 llena; el ratchet debía actuar a las 06:29 y no podía
"""
import pandas as pd
import pytest

import pkg.monkey_bx as mb

COLS = ['symbol', 'orderId', 'side', 'positionSide', 'type', 'origQty', 'price',
        'executedQty', 'avgPrice', 'cumQuote', 'stopPrice', 'time']


def _registro(tmp_path, monkeypatch, filas):
    csv = tmp_path / 'order_id_register.csv'
    pd.DataFrame(filas)[COLS].to_csv(csv, index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'archivos').mkdir(exist_ok=True)
    csv.rename(tmp_path / 'archivos' / 'order_id_register.csv')


def _fila(symbol, hace_min, tipo='STOP_MARKET', stop=100.0):
    ts = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=hace_min)
    return {c: '' for c in COLS} | {
        'symbol': symbol, 'orderId': f'o-{symbol}', 'type': tipo, 'stopPrice': stop,
        'time': int(ts.timestamp() * 1000),
    }


@pytest.mark.parametrize('hace_min', [2, 30, 120, 480])
def test_un_stop_reciente_YA_entra_al_job(tmp_path, monkeypatch, hace_min):
    """La regresión: con el parche de 9 h, nada de esto pasaba el filtro."""
    _registro(tmp_path, monkeypatch, [_fila('BNB-USDT', hace_min)])
    out = mb.filtrando_posiciones_antiguas()
    assert list(out.symbol) == ['BNB-USDT'], f'un stop de {hace_min} min quedó fuera'


def test_el_caso_real_de_bnb(tmp_path, monkeypatch):
    """SL recreado por el BE hace 7,7 h: con el parche quedaba ciego 1,3 h más."""
    _registro(tmp_path, monkeypatch, [_fila('BNB-USDT', 7.7 * 60)])
    assert list(mb.filtrando_posiciones_antiguas().symbol) == ['BNB-USDT']


def test_sigue_excluyendo_lo_recien_creado(tmp_path, monkeypatch):
    """El margen de 1 min existe para no competir con una orden recién puesta."""
    _registro(tmp_path, monkeypatch, [_fila('BNB-USDT', 0.2)])
    assert mb.filtrando_posiciones_antiguas().empty


def test_solo_stop_market(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, [_fila('BNB-USDT', 60, tipo='LIMIT')])
    assert mb.filtrando_posiciones_antiguas().empty


def test_un_stop_por_simbolo(tmp_path, monkeypatch):
    _registro(tmp_path, monkeypatch, [_fila('BNB-USDT', 60, stop=771.14),
                                      _fila('BNB-USDT', 30, stop=752.73)])
    out = mb.filtrando_posiciones_antiguas()
    assert len(out) == 1


def test_sin_archivo_devuelve_vacio_no_rompe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'archivos').mkdir(exist_ok=True)
    out = mb.filtrando_posiciones_antiguas()
    assert out.empty and 'symbol' in out.columns


def test_no_queda_ningun_parche_horario_en_el_filtro():
    """Guard de regresión: que no vuelva a aparecer un offset de husos acá.

    Mira sólo líneas de CÓDIGO — los comentarios documentan el bug y lo nombran.
    """
    import inspect
    codigo = [ln.split('#')[0] for ln in
              inspect.getsource(mb.filtrando_posiciones_antiguas).splitlines()]
    codigo = '\n'.join(codigo)
    assert 'timedelta(hours=' not in codigo, 'volvió un offset horario al filtro'
    assert 'utcnow' in codigo, 'el "ahora" debe ser UTC explícito'
