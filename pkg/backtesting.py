"""
Backtesting toolkit for TRobot.
Features:
* Loads 5‑minute candles, resamples to 30‑minute.
* Calculates indicators (EMA, RSI, ATR, ADX, relative volume).
* Generates trading signals and simulates trades with realistic capital
  management (fixed stake per trade, TP/SL, slippage, equity updates).
* Performs grid‑search optimisation and basic reporting.

2025‑07‑19 – Refactor: logging, capital debit/credit, simple ADX fallback,
NaN‑safe indicator rows, headless‑safe plotting.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

# Opcional: importa talib si usas indicadores técnicos
try:
    import talib
except ImportError:
    talib = None

import vectorbt as vbt
import matplotlib.pyplot as plt
from itertools import product


# Nueva función de backtest con vectorbt
def backtest_with_vectorbt(parametros):
    # Carga de velas históricas
    df = cargar_velas()
    df = df.set_index('date')
    # Pivot de cierres por símbolo
    close_panel = df.pivot(columns='symbol', values='close')

    # Pivot de altos y bajos para ATR/ADX
    high_panel = df.pivot(columns='symbol', values='high')
    low_panel  = df.pivot(columns='symbol', values='low')

    # — Indicadores con índice preservado —
    def _series(fn, panel, **kw):
        "Aplica función TA‑Lib y conserva el índice como Series"
        return panel.apply(lambda col: pd.Series(fn(col, **kw), index=panel.index))

    # --- Indicadores ---
    if talib:
        ema_s = _series(talib.EMA, close_panel, timeperiod=parametros['ema_s'])
        ema_l = _series(talib.EMA, close_panel, timeperiod=parametros['ema_l'])
        rsi   = _series(talib.RSI, close_panel, timeperiod=parametros['rsi'])

        atr = pd.DataFrame({
            sym: pd.Series(
                talib.ATR(high_panel[sym], low_panel[sym], close_panel[sym],
                          timeperiod=parametros['atr']),
                index=close_panel.index
            ) for sym in close_panel.columns
        })
        adx = pd.DataFrame({
            sym: pd.Series(
                talib.ADX(high_panel[sym], low_panel[sym], close_panel[sym],
                          timeperiod=parametros['adx']),
                index=close_panel.index
            ) for sym in close_panel.columns
        })
    else:
        ema_s = vbt.EMA.run(close_panel,  window=parametros['ema_s']).ema
        ema_l = vbt.EMA.run(close_panel,  window=parametros['ema_l']).ema
        rsi   = vbt.RSI.run(close_panel,  window=parametros['rsi']).rsi
        atr   = vbt.ATR.run(high_panel, low_panel, close_panel,
                            window=parametros['atr']).atr
        adx   = vbt.ADX.run(high_panel, low_panel, close_panel,
                            window=parametros['adx']).adx

    # Señales vectorizadas
    entries = (ema_s > ema_l) & (rsi < parametros['rsi_long']) & (adx > parametros['adx_min'])
    exits   = (ema_s < ema_l) | (rsi > parametros['rsi_short'])

    # Distancias de stop‑loss y take‑profit basadas en ATR
    sl_dist = atr * parametros.get('sl_mult', 2)
    tp_dist = atr * parametros.get('tp_mult', 15)

    # --- Backtest con vectorbt (manejo de compatibilidad fill_price) ---
    pfsig_kwargs = dict(
        close=close_panel,
        entries=entries,
        exits=exits,
        init_cash=parametros.get('capital_inicial', 300),
        fees=parametros.get('fee_pct', 0.0007),          # comisión por lado
        slippage=parametros.get('slippage_pct', 0.0005), # 0.05 % de slippage aproximado
        sl_stop=sl_dist,
        tp_stop=tp_dist,
        freq='5T'
    )
    try:
        pf = vbt.Portfolio.from_signals(**pfsig_kwargs, fill_price='close')
    except TypeError:
        # vectorbt < 0.26 no tiene argumento fill_price
        logger.warning("vectorbt sin soporte para fill_price → usando valores por defecto.")
        pf = vbt.Portfolio.from_signals(**pfsig_kwargs)

    # Estadísticas individuales por símbolo
    for sym in close_panel.columns:
        stats = pf.stats(column=sym)
        print(f"\nStats for {sym}:")
        print(stats.loc[['Total Return [%]', 'Win Rate [%]', 'Sharpe Ratio']])
    return pf

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ruta a los archivos de velas
BASE_DIR = Path(__file__).resolve().parent
PATH_5M_LONG = BASE_DIR.parent / "archivos" / "cripto_price_5m_long.csv"

# Carpeta de salida para resultados de backtesting
BACKTEST_DIR = BASE_DIR.parent / "archivos" / "backtesting"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

def cargar_velas(simbolos=None, desde=None, hasta=None):
    """Carga velas desde CSV; opcionalmente filtra por rango y símbolos."""
    df = pd.read_csv(PATH_5M_LONG)
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Filtra por fechas si se requiere
    if desde:
        df = df[df['date'] >= pd.to_datetime(desde)]
    if hasta:
        df = df[df['date'] <= pd.to_datetime(hasta)]

    # Filtra símbolos si se requiere
    if simbolos:
        df = df[df['symbol'].isin(simbolos)]

    return df.sort_values(['symbol', 'date']).reset_index(drop=True)

# Resamplea un DataFrame de velas de 5 minutos a 30 minutos
def resample_5m_to_30m(df_5m):
    """Resamplea velas de 5 minutos a 30 minutos agrupando por símbolo."""
    # Evita SettingWithCopyWarning asegurando copia independiente
    df_5m = df_5m.copy()
    df_5m.loc[:, 'date'] = pd.to_datetime(df_5m['date'])
    before = len(df_5m)
    df_30m = (
        df_5m
        .set_index('date')
        .groupby('symbol')
        .resample('30min')
        .agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        .dropna()
        .reset_index()
    )
    after = len(df_30m)
    logger.debug(f"Resample 5m→30m | símbolo(s): {df_5m['symbol'].unique()} | input={before} output={after}")
    return df_30m

def calcular_indicadores(df, parametros):
    """Calcula indicadores técnicos EMA, RSI, ATR, ADX en el DataFrame."""
    # Asegúrate de que los datos estén ordenados
    df = df.sort_values('date').copy()

    # EMA corta y larga
    df['EMA_S'] = df['close'].ewm(span=parametros['ema_s'], adjust=False).mean()
    df['EMA_L'] = df['close'].ewm(span=parametros['ema_l'], adjust=False).mean()

    # RSI
    if talib:
        df['RSI'] = talib.RSI(df['close'], timeperiod=parametros['rsi'])
    else:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # Wilder's smoothing for RSI
        avg_gain = gain.ewm(alpha=1/parametros['rsi'], adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/parametros['rsi'], adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # ATR
    if talib:
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=parametros['atr'])
    else:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder's smoothing for ATR
        df['ATR'] = tr.ewm(alpha=1/parametros['atr'], adjust=False).mean()

    # ADX
    if talib:
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=parametros['adx'])
    else:
        # Simple ADX 14 implementation (Wilder)
        high = df['high']
        low = df['low']
        close = df['close']
        # Calcula movimientos direccionales manuales
        delta_high = high.diff()
        delta_low = -low.diff()
        plus_dm = delta_high.where((delta_high > delta_low) & (delta_high > 0), 0).fillna(0)
        minus_dm = delta_low.where((delta_low > delta_high) & (delta_low > 0), 0).fillna(0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/parametros['adx'], adjust=False).mean()  # Wilder’s smoothing
        plus_di = 100 * (plus_dm.ewm(alpha=1/parametros['adx'], adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/parametros['adx'], adjust=False).mean() / atr)
        dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
        df['ADX'] = dx.rolling(parametros['adx']).mean()

    # Descarta filas con NaN en indicadores clave
    df = df.dropna(subset=['EMA_S', 'EMA_L', 'RSI', 'ATR', 'ADX']).reset_index(drop=True)

    return df


# Genera las señales LONG/SHORT/ninguna según los parámetros y condiciones de producción
def generar_senales(df, parametros):
    """Genera señales de trading LONG, SHORT o ninguna según condiciones."""
    df = df.copy()

    # Filtro de tendencia en TF mayor (1h en data 30m)
    htf_span = parametros.get('ema_h', 2)  # 2 barras de 30m ≈ 1h
    df['EMA_H'] = df['close'].ewm(span=htf_span, adjust=False).mean()

    # Umbral dinámico de volumen según volatilidad (ATR)
    window = parametros.get('vol_win', 50)
    atr_mean = df['ATR'].rolling(window).mean()
    dynamic_vol_thr = parametros['vol_thr'] * (df['ATR'] / atr_mean).fillna(1)

    # Si falta la columna Rel_Volume, la calcula como volumen actual / promedio móvil de volumen (ventana configurable)
    if 'Rel_Volume' not in df.columns:
        window = parametros.get('vol_win', 50)
        df['Rel_Volume'] = df['volume'] / df['volume'].rolling(window).mean()

    # Inicializa la columna de señales
    df['signal'] = 0

    # Corte y cierre de EMAs para confirmar entradas
    cross_up = (df['EMA_S'].shift(1) <= df['EMA_L'].shift(1)) & (df['EMA_S'] > df['EMA_L'])
    cross_down = (df['EMA_S'].shift(1) >= df['EMA_L'].shift(1)) & (df['EMA_S'] < df['EMA_L'])

    # Tendencia en TF mayor
    trend_long = (df['EMA_S'] > df['EMA_L']) & (df['EMA_L'] > df['EMA_H'])
    trend_short = (df['EMA_S'] < df['EMA_L']) & (df['EMA_L'] < df['EMA_H'])

    long_cond = (
        trend_long &
        cross_up &
        (df['RSI'] < parametros['rsi_long']) &
        (df['ADX'] > parametros['adx_min']) &
        (df['Rel_Volume'] > dynamic_vol_thr)
    )
    short_cond = (
        trend_short &
        cross_down &
        (df['RSI'] > parametros['rsi_short']) &
        (df['ADX'] > parametros['adx_min']) &
        (df['Rel_Volume'] > dynamic_vol_thr)
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    return df


# Nueva función: Simula trades con control de capital y número máximo de trades
def simular_trades_realista(df_30m, df_5m, parametros, simbolo, capital_inicial=300, max_trades=10):
    """Simula trades con capital limitado, stake fijo, TP/SL y reversa."""
    trades = []
    pos = None  # No hay posición abierta al inicio
    capital_disponible = capital_inicial
    cantidad_trades = 0
    # --- nuevas configuraciones desde parametros ---
    fee_pct = parametros.get('fee_pct', 0.0007)      # comisión total (por abrir + cerrar)
    stake_mode = parametros.get('stake', 'fixed')    # 'fixed' o 'percent'
    stake_value = parametros.get('stake_value', capital_inicial)
    max_trades = parametros.get('max_trades', max_trades)
    risk_usd = parametros.get('risk_usd', 10)  # riesgo fijo permitido por trade

    # --- fallback tick_size y lot_size por defecto para todo el scope ---
    tick_size = parametros.get('tick_size', 0.0001)
    lot_size  = parametros.get('lot_size', 0.001)

    spread_mult = parametros.get('spread_mult', 0.05)  # spread como % ATR
    slip_mult   = parametros.get('slip_mult', 0.10)    # desviación slippage vs ATR

    df_30m = df_30m.copy()
    df_5m = df_5m[df_5m['symbol'] == simbolo].copy()
    df_5m['date'] = pd.to_datetime(df_5m['date'])
    df_5m = df_5m.sort_values('date')
    # Propagar señales de 30m a cada vela de 5m para salida en reversa
    df_5m = pd.merge_asof(
        df_5m.sort_values('date'),
        df_30m[['date', 'signal']].sort_values('date'),
        on='date',
        direction='backward'
    ).rename(columns={'signal': 'signal_30m'})

    for i, row in df_30m.iterrows():
        if cantidad_trades >= max_trades and pos is None:
            # No abrimos nuevas posiciones, pero seguimos monitoreando la existente
            continue

        # Si no hay posición abierta y hay señal, calcula tamaño basado en riesgo fijo
        if pos is None and row['signal'] != 0:
            atr = row['ATR']
            if atr == 0 or np.isnan(atr):
                continue

            # Qty para arriesgar exactamente `risk_usd` con el SL definido
            qty = risk_usd / (atr * parametros['sl_mult'])
            if qty <= 0:
                continue

            # Encuentra la primera vela de 5 m posterior para el precio de entrada
            prox_5m = df_5m[df_5m['date'] > row['date']].head(1)
            if prox_5m.empty:
                continue  # no hay vela siguiente

            raw_open = prox_5m.iloc[0]['open']
            fecha_entrada_real = prox_5m.iloc[0]['date']

            # --- Paso 3: añade spread y slippage dinámicos ---
            spread = atr * spread_mult
            slip   = abs(np.random.normal(0, atr * slip_mult))

            if row['signal'] == 1:          # LONG: compras al ask
                entrada = raw_open + spread / 2 + slip
            else:                           # SHORT: vendes al bid
                entrada = raw_open - spread / 2 - slip

            # --- Paso 2: redondeo a tick y lote ---
            tick_size = parametros.get('tick_size', 0.0001)
            lot_size  = parametros.get('lot_size', 0.001)

            entrada = round(entrada / tick_size) * tick_size
            qty     = max(lot_size, round(qty / lot_size) * lot_size)

            # Recalcula monto tras redondeo
            monto_por_trade = qty * entrada

            tp = entrada + parametros['tp_mult'] * atr if row['signal'] == 1 else entrada - parametros['tp_mult'] * atr
            sl = entrada - parametros['sl_mult'] * atr if row['signal'] == 1 else entrada + parametros['sl_mult'] * atr

            # Redondea TP y SL al tick del exchange
            tp = round(tp / tick_size) * tick_size
            sl = round(sl / tick_size) * tick_size

            # Verifica capital suficiente
            if capital_disponible < monto_por_trade:
                continue

            # Debita el stake del capital disponible
            capital_disponible -= monto_por_trade
            tipo = 'LONG' if row['signal'] == 1 else 'SHORT'
            pos = {
                'tipo': tipo,
                'entrada': entrada,
                'fecha_entrada': fecha_entrada_real,
                'tp': tp,
                'sl': sl,
                'idx_entrada': i,
                'fecha_senal_30m': row['date'],
                'qty': qty,
                'monto': monto_por_trade,
            }
            cantidad_trades += 1
            continue

        # Si hay posición abierta
        if pos:
            # Buscar todas las velas de 5m dentro de este periodo de 30m
            intervalo_inicio = row['date']
            intervalo_fin = intervalo_inicio + pd.Timedelta(minutes=30)
            mask = (df_5m['date'] > intervalo_inicio) & (df_5m['date'] <= intervalo_fin)
            velas_5m = df_5m[mask]  

            salida = None
            razon = None
            fecha_salida = None

            # Vectorized detection de salidas en 5m: reversa, SL y TP
            mask_rev = ((pos['tipo'] == 'LONG') & (velas_5m['signal_30m'] == -1)) | \
                       ((pos['tipo'] == 'SHORT') & (velas_5m['signal_30m'] == 1))
            mask_sl  = ((pos['tipo'] == 'LONG') & (velas_5m['low']  <= pos['sl']))    | \
                       ((pos['tipo'] == 'SHORT') & (velas_5m['high'] >= pos['sl']))
            mask_tp  = ((pos['tipo'] == 'LONG') & (velas_5m['high'] >= pos['tp']))    | \
                       ((pos['tipo'] == 'SHORT') & (velas_5m['low']  <= pos['tp']))
            events = pd.DataFrame({
                'date':   velas_5m['date'],
                'price':  velas_5m['close'],
                'reason': np.where(mask_rev, 'Reversa5m', \
                          np.where(mask_sl, 'SL', \
                          np.where(mask_tp, 'TP', None)))
            })
            events = events.dropna(subset=['reason']).sort_values('date')
            if not events.empty:
                first = events.iloc[0]
                salida = first['price']
                razon = first['reason']
                fecha_salida = first['date']

            # Cierra por señal opuesta en 30m
            if salida is None:
                if (pos['tipo'] == 'LONG' and row['signal'] == -1):
                    salida = row['close']
                    razon = 'Reversa'
                    fecha_salida = row['date']
                elif (pos['tipo'] == 'SHORT' and row['signal'] == 1):
                    salida = row['close']
                    razon = 'Reversa'
                    fecha_salida = row['date']

            if salida is not None:
                # --- Ajusta precio de salida por spread y slippage ---
                atr_curr = row['ATR']
                spread_exit = atr_curr * spread_mult
                slip_exit   = abs(np.random.normal(0, atr_curr * slip_mult))

                if pos['tipo'] == 'LONG':
                    salida = salida - spread_exit / 2 - slip_exit
                else:  # SHORT cierra comprando
                    salida = salida + spread_exit / 2 + slip_exit

                # Redondea al tick del exchange
                salida = round(salida / tick_size) * tick_size
                # Calcular qty y ganancia_usd
                qty = pos['qty']
                if pos['tipo'] == 'LONG':
                    ganancia_usd = (salida - pos['entrada']) * qty
                else:  # SHORT
                    ganancia_usd = (pos['entrada'] - salida) * qty
                # aplica fees / slippage
                fee = (pos['entrada'] * qty + salida * qty) * fee_pct
                ganancia_usd -= fee
                # Devuelve stake + P/L al capital
                capital_disponible += pos['monto'] + ganancia_usd
                trades.append({
                    'simbolo': simbolo,
                    'tipo': pos['tipo'],
                    'fecha_entrada_30m': pos['fecha_senal_30m'],
                    'fecha_entrada_real': pos['fecha_entrada'],
                    'entrada': pos['entrada'],
                    'fecha_salida': fecha_salida,
                    'salida': salida,
                    'razon': razon,
                    'ganancia_usd': ganancia_usd,
                    'qty': qty,
                    'monto': pos['monto'], 
                    'duracion_min': (fecha_salida - pos['fecha_entrada']).total_seconds() / 60 if fecha_salida and pos['fecha_entrada'] else None,
                    'tp': pos['tp'],
                    'sl': pos['sl'],
                    'idx_entrada': pos['idx_entrada'],
                    'retorno_usd': ganancia_usd
                })
                pos = None  # Cierra posición

    return pd.DataFrame(trades)


# Analiza los resultados de los trades simulados
def analizar_resultados(trades):
    """Analiza resultados: total, winrate, profit factor, drawdown, retornos."""
    if trades.empty:
        return {
            'total_trades': 0,
            'winrate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'retorno_total_usd': 0,
            'retorno_promedio_usd': 0,
            'mejor_trade': 0,
            'peor_trade': 0
        }
    total_trades = len(trades)
    ganadores = trades[trades['ganancia_usd'] > 0]
    perdedores = trades[trades['ganancia_usd'] < 0]
    winrate = len(ganadores) / total_trades * 100
    profit_factor = ganadores['ganancia_usd'].sum() / abs(perdedores['ganancia_usd'].sum()) if not perdedores.empty else np.inf
    retorno_total_usd = trades['ganancia_usd'].sum()
    retorno_promedio_usd = trades['ganancia_usd'].mean()
    mejor_trade = trades['ganancia_usd'].max()
    peor_trade = trades['ganancia_usd'].min()
    # Equity curve para drawdown
    equity_curve = trades['ganancia_usd'].cumsum()
    max_drawdown = (equity_curve.cummax() - equity_curve).max()
    return {
        'total_trades': total_trades,
        'winrate': winrate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'retorno_total_usd': retorno_total_usd,
        'retorno_promedio_usd': retorno_promedio_usd,
        'mejor_trade': mejor_trade,
        'peor_trade': peor_trade
    }


 
# --------------------------------------------------
#  Grid‑search simple para optimizar parámetros
# --------------------------------------------------
def optimize_parameters(base_cfg, param_grid):
    """
    Recorre todas las combinaciones de param_grid, ejecuta un backtest
    con cada configuración y devuelve la que entregue mejor retorno
    promedio (%) entre todos los símbolos.
    Además exporta un CSV con los resultados de cada intento.
    """
    combos      = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results     = []
    best_cfg    = None
    best_ret    = float("-inf")

    for combo in combos:
        cfg = base_cfg.copy()
        cfg.update(dict(zip(param_names, combo)))

        # Backtest
        pf = backtest_with_vectorbt(cfg)

        # Métrica: retorno medio entre símbolos
        rets = []
        for sym in pf.wrapper.columns:
            stats = pf.stats(column=sym)
            rets.append(stats.loc['Total Return [%]'])
        mean_ret = np.mean(rets)

        results.append({**dict(zip(param_names, combo)),
                        'mean_total_return_%': mean_ret})

        if mean_ret > best_ret:
            best_ret = mean_ret
            best_cfg = cfg.copy()

    # Guarda grid completo
    df = pd.DataFrame(results)
    df.to_csv(BACKTEST_DIR / 'gridsearch_results.csv', index=False)
    logger.info(f"Mejor retorno medio {best_ret:.2f}% con cfg={best_cfg}")
    # Guarda la mejor configuración para uso en producción (dentro de pkg)
    cfg_path = Path(__file__).resolve().parent / 'best_cfg.json'
    with open(cfg_path, 'w') as fp:
        json.dump(best_cfg, fp, indent=2)
    logger.info(f"best_cfg.json guardado en {cfg_path}")
    return best_cfg, df


def main():
    parametros = {
        'ema_s': 16,
        'ema_l': 50,
        'rsi': 14,
        'atr': 14,
        'adx': 14,
        'rsi_long': 67,
        'rsi_short': 33,
        'adx_min': 23,
        'vol_thr': 0.58,
        'tp_mult': 3,          # múltiplo de ATR para TP (más realista)
        'sl_mult': 2,
        'fee_pct': 0.0007,
        'slippage_pct': 0.0005,
        'vol_win': 50,
        'max_trades': 10,
        'stake': 'fixed',
        'stake_value': 30,
    }

    # --------------------------------------------------
    #  Activar grid‑search
    # --------------------------------------------------
    OPTIMIZE = True  # ponlo en False para saltar la búsqueda
    if OPTIMIZE:
        param_grid = {
            'ema_s': [10, 20, 30],
            'ema_l': [50, 100],
            'rsi': [14],
            'rsi_long': [60, 67],
        }
        parametros, _ = optimize_parameters(parametros, param_grid)

    # Ejecutar backtest optimizado con vectorbt
    pf = backtest_with_vectorbt(parametros)

    # --- Resumen legible por símbolo ---
    symbols = pf.wrapper.columns
    rows = []
    for sym in symbols:
        stats = pf.stats(column=sym)
        rows.append({
            'symbol': sym,
            'Total Return [%]': stats.loc['Total Return [%]'],
            'Win Rate [%]': stats.loc['Win Rate [%]'],
            'Sharpe Ratio': stats.loc['Sharpe Ratio'],
        })
    human = pd.DataFrame(rows).set_index('symbol')
    human = human.sort_values(by='Total Return [%]', ascending=False)

    # ---- Exportar e imprimir resumen ----
    summary_path = BACKTEST_DIR / "summary_per_symbol.csv"
    human.to_csv(summary_path)
    logger.info(f"Resumen por símbolo guardado en {summary_path}")
    print("\nResumen por símbolo:\n")
    print(human.to_string())

    # --- Datos para gráfico compuesto ---
    plot_possible = True
    # Extrae trades de vectorbt
    trades_rec = pf.trades.records_readable

    # -------- Mapear símbolo para cada trade --------
    def infer_symbol_map(trades_df, pf_obj):
        """Devuelve trades_df con columna 'symbol' garantizada."""
        # 1) Ya existe
        if 'symbol' in trades_df.columns:
            return trades_df
        # 2) Columnas comunes en versiones vbt
        for cand in ['column', 'col', 'column_idx', 'col_idx']:
            if cand in trades_df.columns:
                raw = trades_df[cand]
                # numérico → índice de columna
                if pd.api.types.is_integer_dtype(raw) or pd.api.types.is_float_dtype(raw):
                    trades_df['symbol'] = raw.astype(int).map(lambda i: pf_obj.wrapper.columns[int(i)])
                else:
                    trades_df['symbol'] = raw
                return trades_df
        # 3) Mirar pf.trades.records si es ndarray estructurado
        recs = pf_obj.trades.records
        if not isinstance(recs, pd.DataFrame):
            # ndarray con dtype names
            if 'col' in recs.dtype.names:
                trades_df['symbol'] = [pf_obj.wrapper.columns[int(i)] for i in recs['col']]
                return trades_df
            if 'column' in recs.dtype.names:
                trades_df['symbol'] = recs['column']
                return trades_df
        else:
            # DataFrame: intenta mismo set de candidatos
            for cand in ['column', 'col', 'column_idx', 'col_idx']:
                if cand in recs.columns:
                    raw = recs[cand]
                    if pd.api.types.is_integer_dtype(raw):
                        trades_df['symbol'] = raw.map(lambda i: pf_obj.wrapper.columns[int(i)])
                    else:
                        trades_df['symbol'] = raw
                    return trades_df
        raise KeyError("No se pudo inferir la columna 'symbol' en trades records")

    trades_rec = infer_symbol_map(trades_rec, pf)

    # ---------- Localizar / derivar retorno porcentual ----------
    candidate_cols = [
        'pnl_%', 'pnl_pct', 'pnl_percent',
        'return_%', 'return_pct', 'return_percent', 'return', 'Return',
        'pnl'
    ]
    pnl_col = None
    for cand in candidate_cols:
        if cand in trades_rec.columns:
            pnl_col = cand
            break

    if pnl_col is None or pnl_col == 'pnl':
        # Calcular % manual si solo hay pnl absoluto
        if {'pnl', 'entry_price', 'size'}.issubset(trades_rec.columns):
            trades_rec['pnl_perc'] = (
                trades_rec['pnl'] /
                (trades_rec['entry_price'].abs() * trades_rec['size'].abs())
            ) * 100
            pnl_col = 'pnl_perc'
        else:
            logger.warning(
                "No se encontró columna de retorno porcentual ni "
                "datos suficientes para derivarla. Se omitirá el gráfico."
            )
            plot_possible = False

    if plot_possible:
        fig, ax = plt.subplots()
        ax.bar(human.index.astype(str), human['Total Return [%]'])
        ax.set_ylabel('Retorno total (%)')
        ax.set_title('Retorno total por símbolo')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = BACKTEST_DIR / "retorno_por_simbolo.png"
        plt.savefig(output_path)
        plt.close(fig)
        logger.info(f"Gráfico guardado en {output_path}")
        # ---- Exportar trades detallados para comparación con P&L real ----
        # Construir DataFrame directamente desde pf.trades.records para asegurar
        # la presencia de los campos necesarios.
        rec_df = pd.DataFrame(pf.trades.records)

        # Añadir símbolo a partir del índice de columna ('col' → nombre del símbolo)
        rec_df['symbol'] = rec_df['col'].apply(lambda i: pf.wrapper.columns[int(i)])

        # Derivar timestamps de entrada y salida usando el índice de precios
        idx_ts = pf.wrapper.index
        rec_df['entry_time'] = rec_df['entry_idx'].apply(lambda i: idx_ts[int(i)])
        rec_df['exit_time']  = rec_df['exit_idx'].apply(lambda i: idx_ts[int(i)])

        # Renombrar precios y PnL para estandarizar
        rec_df = rec_df.rename(columns={
            'entry_price': 'entry_price',
            'exit_price':  'exit_price',
            'pnl':         'pnl_usd'
        })

        # Seleccionar y re‑ordenar columnas finales
        export_cols = ['symbol', 'entry_time', 'exit_time',
                       'entry_price', 'exit_price', 'pnl_usd']
        trades_std = rec_df[export_cols]

        std_path = BACKTEST_DIR / "sim_trades.csv"
        trades_std.to_csv(std_path, index=False)
        logger.info(f"Trades simulados exportados a {std_path}")


if __name__ == "__main__":
    main()