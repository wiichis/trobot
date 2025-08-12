def resumen_humano(pf, out_dir=None):
    """Genera un resumen corto "para humanos" en consola y en un TXT.
    Devuelve también el DataFrame por símbolo ordenado por retorno.
    """
    out_dir = BACKTEST_DIR if out_dir is None else out_dir
    symbols = list(pf.wrapper.columns)

    # --- Tabla por símbolo (ordenada) ---
    rows = []
    for sym in symbols:
        stats = pf.stats(column=sym)
        rows.append({
            'symbol': sym,
            'Total Return [%]': float(stats.loc['Total Return [%]']),
            'Win Rate [%]': float(stats.loc['Win Rate [%]']),
            'Sharpe Ratio': float(stats.loc['Sharpe Ratio']),
        })
    human = pd.DataFrame(rows).set_index('symbol').sort_values(by='Total Return [%]', ascending=False)

    # --- Agregados simples ---
    try:
        rec = pf.trades.records_readable
        total_trades = int(len(rec))
        gains = rec[rec['pnl'] > 0]['pnl'].sum()
        losses = -rec[rec['pnl'] < 0]['pnl'].sum()
        profit_factor = float(gains / losses) if losses > 0 else float('inf')
        # Duración media aprox. en minutos (cada barra = 5m)
        if {'entry_idx', 'exit_idx'}.issubset(rec.columns):
            avg_hold_min = float(((rec['exit_idx'] - rec['entry_idx']) * 5).mean())
        else:
            avg_hold_min = float('nan')
    except Exception:
        total_trades = 0
        profit_factor = float('nan')
        avg_hold_min = float('nan')

    winrate_mean = float(np.nanmean([pf.stats(column=s).loc['Win Rate [%]'] for s in symbols]))
    total_return_mean = float(np.nanmean([pf.stats(column=s).loc['Total Return [%]'] for s in symbols]))
    sharpe_mean = float(np.nanmean([pf.stats(column=s).loc['Sharpe Ratio'] for s in symbols]))

    # --- Texto simple y claro ---
    top_symbol = human.index[0] if not human.empty else 'N/A'
    top_ret = human.iloc[0]['Total Return [%]'] if not human.empty else float('nan')

    lines = [
        "\n════════ Resumen del Backtest ════════",
        f"Símbolos evaluados: {len(symbols)}",
        f"Trades cerrados: {total_trades}",
        f"Winrate medio: {winrate_mean:.1f}% | Profit Factor: {profit_factor:.2f}",
        f"Retorno medio: {total_return_mean:.1f}% | Sharpe medio: {sharpe_mean:.2f}",
        f"Duración media por trade: {avg_hold_min:.1f} min" if not np.isnan(avg_hold_min) else "Duración media por trade: N/D",
        f"Mejor símbolo: {top_symbol} ({top_ret:.1f}%)" if not human.empty else "Mejor símbolo: N/D",
        "────────────────────────────────────",
        "Top por retorno (%)\n" + human['Total Return [%]'].round(2).to_string(),
    ]
    text = "\n".join(lines) + "\n"

    # Imprime en consola y guarda en TXT
    print(text)
    out_path = out_dir / 'resumen.txt'
    try:
        with open(out_path, 'w') as f:
            f.write(text)
        logger.info(f"Resumen humano guardado en {out_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar resumen humano: {e}")

    return human
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

# --- Opciones de ejecución rápida ---
QUICK_TEST = False   # True para test rápido, False para test general
FAST_OPT = True    # True para grid de optimización reducido (acelera el backtest)

# Refinamiento en dos etapas (coarse → fine)
COARSE_FINE = True   # activa la segunda pasada de optimización refinada cuando FAST_OPT=True
TOP_N_REFINE = 5     # número de filas TOP del grid rápido a usar para el refinamiento

# Límites globales de optimización
MAX_COMBOS = 10_000     # tope máximo de combinaciones a evaluar
RNG_SEED   = 42         # semilla para muestreo reproducible
OPT_SYMBOLS_LIMIT = 8   # límite de símbolos durante optimize (solo optimización)

# Límites y control del grid refinado (fine)
REFINE_KEYS = ['ema_s','ema_l','adx_min','rsi_long','rsi_short','vol_thr','rel_v_5m_factor']
REFINE_DELTAS_SCALE = 1   # usa vecindario ±1*step (no ±2) para reducir combinaciones
COARSE_FINE_MAX_COMBOS = 10_000  # tope duro de combinaciones en la segunda pasada

# Construye un grid refinado alrededor de los mejores valores del grid coarse
def _refine_grid_from_top(top_df, keys):
    import numpy as np
    from math import prod
    # 1) Filtrar keys a refinar
    keys = [k for k in keys if k in REFINE_KEYS]
    grid = {}
    for k in keys:
        if k not in top_df.columns:
            continue
        vals = [x for x in top_df[k].dropna().tolist()]
        if not vals:
            continue
        # Normaliza numéricos
        norm_vals = []
        for x in vals:
            try:
                xv = float(x)
                norm_vals.append(int(xv) if abs(xv - int(xv)) < 1e-9 else round(xv, 4))
            except Exception:
                norm_vals.append(x)
        uniq = sorted(set(norm_vals))
        vmin, vmax = (min(uniq), max(uniq)) if all(isinstance(x, (int, float)) for x in uniq) else (None, None)
        if vmin is None:
            grid[k] = uniq
            continue
        # pasos por parámetro
        if k in ('ema_s',):
            step = 2;  lb, ub = 5, 30
        elif k in ('ema_l',):
            step = 10; lb, ub = 20, 120
        elif k in ('rsi_long', 'rsi_short'):
            step = 2;  lb, ub = 20, 80
        elif k == 'adx_min':
            step = 1;  lb, ub = 5, 30
        elif k == 'rsi_lb':
            step = 1;  lb, ub = 2, 12
        elif k == 'vol_thr':
            step = 0.05; lb, ub = 0.05, 1.0
        elif k == 'rel_v_5m_factor':
            step = 0.2;  lb, ub = 1.0, 3.0
        else:
            step = 1;  lb, ub = vmin, vmax
        # 2) Vecindario reducido: ±scale*step
        scale = max(1, int(REFINE_DELTAS_SCALE))
        deltas = (-scale*step, 0, scale*step)
        candidates = set()
        for v in uniq:
            for d in deltas:
                nv = v + d
                if isinstance(step, float):
                    nv = round(nv, 4)
                else:
                    nv = int(round(nv))
                if nv < lb or nv > ub:
                    continue
                candidates.add(nv)
        grid[k] = sorted(candidates)
    # 3) Tope duro de combinaciones: recorta centrando valores
    def _center_crop(vals, keep):
        if len(vals) <= keep:
            return vals
        mid = len(vals)//2
        half = keep//2
        start = max(0, mid - half)
        return vals[start:start+keep]
    def _total(grid):
        sizes = [len(v) for v in grid.values() if len(v)>0]
        return prod(sizes) if sizes else 0
    total = _total(grid)
    if total > COARSE_FINE_MAX_COMBOS:
        while total > COARSE_FINE_MAX_COMBOS:
            biggest = sorted(grid.items(), key=lambda kv: len(kv[1]), reverse=True)
            k, vals = biggest[0]
            new_vals = _center_crop(vals, 3)
            if len(new_vals) == len(vals):
                break
            grid[k] = new_vals
            total = _total(grid)
    return grid

# Opcional: importa talib si usas indicadores técnicos
try:
    import talib
except ImportError:
    talib = None

import vectorbt as vbt
import matplotlib.pyplot as plt
from itertools import product
import sys

# --- Generador para limitar combinaciones y muestrear aleatoriamente si excede el tope ---
from math import prod

def _sample_param_combos(param_grid, max_combos=MAX_COMBOS, seed=RNG_SEED):
    """Genera combinaciones de parámetros. Si el producto total supera max_combos,
    usa muestreo aleatorio uniforme por parámetro (reproducible) hasta max_combos."""
    keys = list(param_grid.keys())
    sizes = [len(param_grid[k]) for k in keys]
    total = int(prod(sizes)) if sizes else 0
    if total == 0:
        return keys, []
    if total <= max_combos:
        combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]
        return keys, combos
    # RandomizedSearch
    logger.warning(f"Grid con {total} combinaciones > MAX_COMBOS={max_combos}. Usando muestreo aleatorio.")
    import numpy as np
    rng = np.random.RandomState(seed)
    n_iter = min(max_combos, total)
    combos, seen = [], set()
    attempts, max_attempts = 0, n_iter * 10
    while len(combos) < n_iter and attempts < max_attempts:
        attempts += 1
        cfg = {k: rng.choice(param_grid[k]) for k in keys}
        key = tuple(cfg[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        combos.append(cfg)
    if len(combos) < n_iter:
        logger.warning(f"Se generaron {len(combos)} combinaciones únicas (menos de {n_iter}) por duplicados.")
    return keys, combos


# Modo de detalle en consola (se desactiva durante la optimización)
DETAILS = True

def _progress(i, total, prefix='Optimizando', length=30):
    """Imprime una barra de progreso en una sola línea."""
    if total <= 0:
        return
    pct = (i + 1) / total
    filled = int(length * pct)
    bar = '#' * filled + '-' * (length - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {pct*100:5.1f}% ({i+1}/{total})")
    sys.stdout.flush()
    if i + 1 == total:
        sys.stdout.write("\n")


# Nueva función de backtest con vectorbt
def backtest_with_vectorbt(parametros):
    # Carga de velas históricas
    df = cargar_velas()
    opt_limit = int(parametros.get('opt_symbols_limit', 0) or 0)
    if opt_limit > 0:
        try:
            keep = (
                df.groupby('symbol')['volume'].mean().sort_values(ascending=False).head(opt_limit).index.tolist()
            )
            df = df[df['symbol'].isin(keep)]
            logger.info(f"Optimizando con top {len(keep)} símbolos por volumen: {keep}")
        except Exception as e:
            logger.warning(f"No se pudo limitar símbolos para optimize: {e}")
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

    # Lookback de RSI en 5m para alinear con la lógica de 30m
    rsi_lb = int(parametros.get('rsi_lb', 3))
    rsi_min = rsi.rolling(window=rsi_lb, min_periods=1).min()
    rsi_max = rsi.rolling(window=rsi_lb, min_periods=1).max()

    # --- Señales vectorizadas alineadas con reglas de producción ---
    # EMA de TF mayor (aprox 1h en datos de 5m → 12 barras por defecto)
    ema_h_win = parametros.get('ema_h_5m', 12)
    if talib:
        ema_h = _series(talib.EMA, close_panel, timeperiod=ema_h_win)
    else:
        ema_h = vbt.EMA.run(close_panel, window=ema_h_win).ema

    # Volumen relativo y umbral dinámico por volatilidad
    volume_panel = df.pivot(columns='symbol', values='volume')
    vol_win = parametros.get('vol_win', 50)
    rel_volume = volume_panel / volume_panel.rolling(vol_win).mean()
    atr_mean = atr.rolling(vol_win).mean()
    dynamic_vol_thr = parametros.get('vol_thr', 1.0) * (atr / atr_mean).fillna(1)

    # Cruces y tendencias
    cross_up = (ema_s.shift(1) <= ema_l.shift(1)) & (ema_s > ema_l)
    cross_down = (ema_s.shift(1) >= ema_l.shift(1)) & (ema_s < ema_l)
    trend_long = (ema_s > ema_l) & (ema_h > ema_l)
    trend_short = (ema_s < ema_l) & (ema_h < ema_l)

    long_entries = (cross_up) & (rsi_min < parametros['rsi_long'])
    long_exits = (rsi_max > parametros['rsi_long'])

    short_entries = (cross_down) & (rsi_max > parametros['rsi_short'])
    short_exits = (rsi_min < parametros['rsi_short'])
    
    # --- Alineación/booleanos + diagnóstico de señales ---
    long_entries = long_entries.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
    long_exits   = long_exits.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
    short_entries = short_entries.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
    short_exits   = short_exits.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)

    overlap = ((long_entries & long_exits) | (short_entries & short_exits)).sum().sum()
    logger.info(f"Solapes entry/exit en misma vela: {int(overlap)}")

    l_count = int(long_entries.sum().sum())
    s_count = int(short_entries.sum().sum())
    logger.info(f"Signals (pre‑fallback): long={l_count}, short={s_count}")
    # Controlar fallback vía parámetros (por defecto activo solo en QUICK_TEST)
    allow_fallback = bool(parametros.get('allow_fallback', QUICK_TEST))

    # Si no hay ninguna señal, relaja filtros para validar funcionamiento base
    if (l_count + s_count) == 0 and allow_fallback:
        logger.warning("Sin señales con filtros actuales; relajando filtros (sin ADX/volumen/EMA_H).")
        long_entries  = ( (ema_s.shift(1) <= ema_l.shift(1)) & (ema_s > ema_l) )
        long_exits    = ( (ema_s.shift(1) >= ema_l.shift(1)) & (ema_s < ema_l) )
        short_entries = ( (ema_s.shift(1) >= ema_l.shift(1)) & (ema_s < ema_l) )
        short_exits   = ( (ema_s.shift(1) <= ema_l.shift(1)) & (ema_s > ema_l) )

        long_entries = long_entries.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
        long_exits   = long_exits.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
        short_entries = short_entries.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)
        short_exits   = short_exits.reindex(index=close_panel.index, columns=close_panel.columns).fillna(False).astype(bool)

        l_count = int(long_entries.sum().sum())
        s_count = int(short_entries.sum().sum())
        logger.info(f"Signals (post‑fallback): long={l_count}, short={s_count}")

    # Distancias de stop basadas en ATR (vectorizadas)

    # --- Tamaño de posición basado en stake USD (permite fraccional) ---
    stake_usd = float(parametros.get('stake_value', 30))
    # tamaño en "unidades" = USD / precio; permite fraccional
    size_df = (stake_usd / close_panel).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Diagnóstico rápido de tamaños
    try:
        sz_min, sz_med, sz_max = size_df.min().min(), size_df.median().median(), size_df.max().max()
        logger.info(f"Sizing stake=${stake_usd} → size[min/med/max]={sz_min:.6f}/{sz_med:.6f}/{sz_max:.6f}")
    except Exception:
        pass

    # --- Backtest con vectorbt (precio de llenado, sin stops automáticos aquí) ---
    pfsig_kwargs = dict(
        close=close_panel,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=size_df,
        init_cash=parametros.get('capital_inicial', 300),
        fees=parametros.get('fee_pct', 0.0007),
        slippage=parametros.get('slippage_pct', 0.0005),
        freq='5T'
    )
    try:
        pf = vbt.Portfolio.from_signals(**pfsig_kwargs, fill_price='close')
    except TypeError:
        pf = vbt.Portfolio.from_signals(**pfsig_kwargs)

    # --- Diagnóstico: conteo de trades ---
    try:
        recs = pf.trades.records
        n_trades = len(recs) if hasattr(recs, '__len__') else 0
        logger.info(f"vbt trades cerrados: {n_trades}")
    except Exception as e:
        logger.warning(f"No se pudo leer trades de vectorbt: {e}")

    # Estadísticas individuales por símbolo
    if DETAILS:
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

    debug_sig = bool(parametros.get('debug_signals', QUICK_TEST))

    # Filtro de tendencia en TF mayor (1h en data 30m)
    htf_span = parametros.get('ema_h', 2)  # 2 barras de 30m ≈ 1h
    df['EMA_H'] = df['close'].ewm(span=htf_span, adjust=False).mean()
    # Lookback de RSI para permitir que la condición se cumpla en las últimas N velas
    rsi_lb = int(parametros.get('rsi_lb', 3))
    rsi_roll_min = df['RSI'].rolling(window=rsi_lb, min_periods=1).min()
    rsi_roll_max = df['RSI'].rolling(window=rsi_lb, min_periods=1).max()

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

    # Tendencia en TF mayor coherente: LONG si EMA_H > EMA_L; SHORT si EMA_H < EMA_L
    trend_long = (df['EMA_S'] > df['EMA_L']) & (df['EMA_H'] > df['EMA_L'])
    trend_short = (df['EMA_S'] < df['EMA_L']) & (df['EMA_H'] < df['EMA_L'])

    # --- Diagnóstico de filtros (conteo de verdaderos) ---
    filt = {}
    filt['cross_up'] = cross_up
    filt['cross_down'] = cross_down
    filt['trend_long'] = trend_long
    filt['trend_short'] = trend_short
    filt['rsi_low']  = (rsi_roll_min < parametros['rsi_long'])
    filt['rsi_high'] = (rsi_roll_max > parametros['rsi_short'])
    window = parametros.get('vol_win', 50)
    atr_mean = df['ATR'].rolling(window).mean()
    dynamic_vol_thr = parametros['vol_thr'] * (df['ATR'] / atr_mean).fillna(1)
    filt['rel_vol_pass'] = (df['Rel_Volume'] > dynamic_vol_thr)
    filt['adx_pass'] = (df['ADX'] > parametros['adx_min'])
    if debug_sig:
        try:
            logger.info(
                'Diag 30m: cross_up=%d, cross_down=%d, trend_long=%d, trend_short=%d, rsi_low=%d, rsi_high=%d, rel_vol=%d, adx=%d',
                int(filt['cross_up'].sum()), int(filt['cross_down'].sum()),
                int(filt['trend_long'].sum()), int(filt['trend_short'].sum()),
                int(filt['rsi_low'].sum()), int(filt['rsi_high'].sum()),
                int(filt['rel_vol_pass'].sum()), int(filt['adx_pass'].sum())
            )
        except Exception:
            pass

    long_cond = (
        filt['trend_long'] &
        filt['cross_up'] &
        filt['rsi_low'] &
        filt['adx_pass'] &
        filt['rel_vol_pass']
    )
    short_cond = (
        filt['trend_short'] &
        filt['cross_down'] &
        filt['rsi_high'] &
        filt['adx_pass'] &
        filt['rel_vol_pass']
    )

    df.loc[long_cond, 'signal'] = 1
    df.loc[short_cond, 'signal'] = -1

    if debug_sig:
        try:
            logger.info('Diag 30m: señales long=%d, short=%d', int(long_cond.sum()), int(short_cond.sum()))
        except Exception:
            pass

    return df


# Nueva función: Simula trades con control de capital y número máximo de trades
def simular_trades_realista(df_30m, df_5m, parametros, simbolo, capital_inicial=300, max_trades=10):
    """Simula trades con capital limitado, stake fijo, TP/SL y reversa."""
    np.random.seed(parametros.get('seed', 42))
    trades = []
    pos = None  # No hay posición abierta al inicio
    capital_disponible = capital_inicial
    cantidad_trades = 0
    discards = []  # logs de señales descartadas con motivo
    # --- nuevas configuraciones desde parametros ---
    fee_pct = parametros.get('fee_pct', 0.0007)      # comisión total (por abrir + cerrar)
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
    # ATR 5m para trailing stop
    atr_win = parametros.get('atr', 14)
    if 'ATR_5m' not in df_5m.columns:
        if talib:
            df_5m['ATR_5m'] = talib.ATR(df_5m['high'], df_5m['low'], df_5m['close'], timeperiod=atr_win)
        else:
            tr1 = df_5m['high'] - df_5m['low']
            tr2 = (df_5m['high'] - df_5m['close'].shift()).abs()
            tr3 = (df_5m['low'] - df_5m['close'].shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df_5m['ATR_5m'] = tr.ewm(alpha=1/atr_win, adjust=False).mean()

    # EMA 5m para micro‑confirmación de entrada
    ema_s_win = parametros.get('ema_s', 16)
    ema_l_win = parametros.get('ema_l', 50)
    if 'EMA_S_5m' not in df_5m.columns or 'EMA_L_5m' not in df_5m.columns:
        if talib:
            df_5m['EMA_S_5m'] = talib.EMA(df_5m['close'], timeperiod=ema_s_win)
            df_5m['EMA_L_5m'] = talib.EMA(df_5m['close'], timeperiod=ema_l_win)
        else:
            df_5m['EMA_S_5m'] = df_5m['close'].ewm(span=ema_s_win, adjust=False).mean()
            df_5m['EMA_L_5m'] = df_5m['close'].ewm(span=ema_l_win, adjust=False).mean()

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
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'no_ATR'})
                continue

            # Qty para arriesgar exactamente `risk_usd` con el SL definido
            qty = risk_usd / (atr * parametros['sl_mult'])
            if qty <= 0:
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'qty_nonpositive'})
                continue

            # Encuentra la primera vela de 5 m posterior para el precio de entrada
            prox_5m = df_5m[df_5m['date'] > row['date']].head(1)
            if prox_5m.empty:
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'no_next_5m'})
                continue  # no hay vela siguiente

            # --- Micro‑filtro 5m para confirmar la entrada ---
            ema_s_5 = float(prox_5m.iloc[0]['EMA_S_5m']) if not pd.isna(prox_5m.iloc[0]['EMA_S_5m']) else None
            ema_l_5 = float(prox_5m.iloc[0]['EMA_L_5m']) if not pd.isna(prox_5m.iloc[0]['EMA_L_5m']) else None
            close_5 = float(prox_5m.iloc[0]['close'])
            # Si aún no hay EMAs (por ventana), salta
            if ema_s_5 is None or ema_l_5 is None:
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'ema5_nan'})
                continue
            if row['signal'] == 1:
                # LONG: solo alineación de EMAs
                if not (ema_s_5 > ema_l_5):
                    discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'microfilter_long_fail'})
                    continue
            else:
                # SHORT: solo alineación de EMAs
                if not (ema_s_5 < ema_l_5):
                    discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'microfilter_short_fail'})
                    continue

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

            entrada = round(entrada / tick_size) * tick_size
            qty     = max(lot_size, round(qty / lot_size) * lot_size)

            # Capear qty por capital disponible (comprar lo que alcance) usando solo 50% del capital disponible
            oper_cap = capital_disponible * 0.5
            max_qty_cap = (oper_cap / entrada) if entrada > 0 else 0
            max_qty_cap = max(0.0, round(max_qty_cap / lot_size) * lot_size)
            if max_qty_cap <= 0:
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'insufficient_capital_zero'})
                continue
            if qty > max_qty_cap:
                qty = max_qty_cap

            # Recalcula monto tras redondeo
            monto_por_trade = qty * entrada

            tp = entrada + parametros['tp_mult'] * atr if row['signal'] == 1 else entrada - parametros['tp_mult'] * atr
            sl = entrada - parametros['sl_mult'] * atr if row['signal'] == 1 else entrada + parametros['sl_mult'] * atr

            # Redondea TP y SL al tick del exchange
            tp = round(tp / tick_size) * tick_size
            sl = round(sl / tick_size) * tick_size

            # --- Garantizar orden lógico tras redondeo ---
            if row['signal'] == 1:  # LONG
                # asegurar sl < entrada < tp; si colisionan, separa 1 tick
                if not (sl < entrada < tp):
                    sl = min(sl, entrada - tick_size)
                    tp = max(tp, entrada + tick_size)
            else:  # SHORT
                # asegurar tp < entrada < sl
                if not (tp < entrada < sl):
                    tp = min(tp, entrada - tick_size)
                    sl = max(sl, entrada + tick_size)

            # Verifica capital suficiente
            if capital_disponible < monto_por_trade:
                discards.append({'simbolo': simbolo, 'date_30m': row['date'], 'reason': 'insufficient_capital', 'monto': float(monto_por_trade), 'capital': float(capital_disponible)})
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
            # Distancia inicial de stop (1R) y flag de break-even
            pos['sl_dist'] = parametros.get('sl_mult', 2) * row['ATR']
            pos['be_set'] = False
            # Estado inicial para trailing por velas de 5m
            pos['last_close'] = entrada
            if tipo == 'LONG':
                pos['trail_peak'] = entrada  # máximo de cierres desde la entrada
            else:
                pos['trail_trough'] = entrada  # mínimo de cierres desde la entrada
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

            # --- Salida intrabar 5m usando OHLC ---
            salida = None
            razon = None
            fecha_salida = None

            # Reglas: para LONG si en la misma vela se tocan TP y SL, asumimos conservador (SL primero).
            # Para SHORT análogo (SL primero).
            for _, v in velas_5m.iterrows():
                # --- Regla de Break-Even (1R) ---
                # Si el precio se mueve a favor al menos 1R desde la entrada, sube/baja el SL a la entrada una sola vez.
                if not pos.get('be_set', False):
                    if pos['tipo'] == 'LONG':
                        moved = (v['high'] - pos['entrada'])
                        if moved >= pos.get('sl_dist', 0):
                            new_sl_be = round(pos['entrada'] / tick_size) * tick_size
                            # Asegura orden lógico sin invadir TP
                            pos['sl'] = min(max(pos['sl'], new_sl_be), pos['tp'] - tick_size)
                            pos['be_set'] = True
                    else:  # SHORT
                        moved = (pos['entrada'] - v['low'])
                        if moved >= pos.get('sl_dist', 0):
                            new_sl_be = round(pos['entrada'] / tick_size) * tick_size
                            pos['sl'] = max(min(pos['sl'], new_sl_be), pos['tp'] + tick_size)
                            pos['be_set'] = True

                # --- Trailing SL por cada vela de 5m (sin lookahead) ---
                # Usa el cierre anterior acumulado (last_close) para evitar mirar dentro de la vela actual.
                atr5 = v.get('ATR_5m', np.nan)
                if pd.isna(atr5) or atr5 == 0:
                    atr5 = row['ATR']  # fallback al ATR de 30m

                if pos['tipo'] == 'LONG':
                    # Actualiza el máximo de cierres hasta la vela previa
                    peak = max(pos.get('trail_peak', pos['entrada']), pos.get('last_close', pos['entrada']))
                    pos['trail_peak'] = peak
                    new_sl = peak - parametros.get('sl_mult', 2) * atr5
                    new_sl = round(new_sl / tick_size) * tick_size
                    # Mantén orden lógico: sl < tp
                    if new_sl > pos['sl']:
                        pos['sl'] = min(new_sl, pos['tp'] - tick_size)
                else:  # SHORT
                    trough = min(pos.get('trail_trough', pos['entrada']), pos.get('last_close', pos['entrada']))
                    pos['trail_trough'] = trough
                    new_sl = trough + parametros.get('sl_mult', 2) * atr5
                    new_sl = round(new_sl / tick_size) * tick_size
                    # Mantén orden lógico: tp < sl
                    if new_sl < pos['sl']:
                        pos['sl'] = max(new_sl, pos['tp'] + tick_size)

                o, h, l, c, t = v['open'], v['high'], v['low'], v['close'], v['date']
                if pos['tipo'] == 'LONG':
                    hit_tp = h >= pos['tp']
                    hit_sl = l <= pos['sl']
                    if hit_tp and hit_sl:
                        # Empate: conservador → SL primero
                        salida = pos['sl']
                        razon = 'SL_samebar'
                        fecha_salida = t
                        break
                    elif hit_sl:
                        salida = pos['sl']
                        razon = 'SL'
                        fecha_salida = t
                        break
                    elif hit_tp:
                        salida = pos['tp']
                        razon = 'TP'
                        fecha_salida = t
                        break
                else:  # SHORT
                    hit_tp = l <= pos['tp']
                    hit_sl = h >= pos['sl']
                    if hit_tp and hit_sl:
                        salida = pos['sl']
                        razon = 'SL_samebar'
                        fecha_salida = t
                        break
                    elif hit_sl:
                        salida = pos['sl']
                        razon = 'SL'
                        fecha_salida = t
                        break
                    elif hit_tp:
                        salida = pos['tp']
                        razon = 'TP'
                        fecha_salida = t
                        break
                # Actualiza el último cierre para el cálculo de trailing en la próxima vela
                pos['last_close'] = v['close']

            # Cierre por señal opuesta en 30m (si no hubo TP/SL en los 5m)
            if salida is None:
                if (pos['tipo'] == 'LONG' and row['signal'] == -1) or (pos['tipo'] == 'SHORT' and row['signal'] == 1):
                    salida = row['close']
                    razon = 'Reversa_30m'
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
                    'retorno_usd': ganancia_usd,
                    'entry_signal_30m': 1 if pos['tipo'] == 'LONG' else -1,
                })
                pos = None  # Cierra posición

    # --- Cierre forzado al final del dataset (EOD) si queda posición abierta ---
    if pos:
        try:
            # usa el último precio disponible de 5m posterior a la última fecha de 30m
            last30 = df_30m['date'].max()
            last5 = df_5m[df_5m['date'] > last30]
            if last5.empty:
                last5 = df_5m.tail(1)
            salida = float(last5.iloc[-1]['close'])
            t_out = last5.iloc[-1]['date']

            # Ajusta por spread/slippage como en salidas normales
            atr_curr = df_30m.iloc[-1]['ATR'] if 'ATR' in df_30m.columns else df_5m.iloc[-1].get('ATR_5m', np.nan)
            if pd.isna(atr_curr) or atr_curr == 0:
                atr_curr = 0.0
            spread_exit = atr_curr * spread_mult
            slip_exit   = abs(np.random.normal(0, atr_curr * slip_mult)) if atr_curr > 0 else 0.0
            if pos['tipo'] == 'LONG':
                salida = salida - spread_exit / 2 - slip_exit
            else:
                salida = salida + spread_exit / 2 + slip_exit
            salida = round(salida / tick_size) * tick_size

            qty = pos['qty']
            if pos['tipo'] == 'LONG':
                ganancia_usd = (salida - pos['entrada']) * qty
            else:
                ganancia_usd = (pos['entrada'] - salida) * qty
            fee = (pos['entrada'] * qty + salida * qty) * fee_pct
            ganancia_usd -= fee
            capital_disponible += pos['monto'] + ganancia_usd
            trades.append({
                'simbolo': simbolo,
                'tipo': pos['tipo'],
                'fecha_entrada_30m': pos['fecha_senal_30m'],
                'fecha_entrada_real': pos['fecha_entrada'],
                'entrada': pos['entrada'],
                'fecha_salida': t_out,
                'salida': salida,
                'razon': 'EOD',
                'ganancia_usd': ganancia_usd,
                'qty': qty,
                'monto': pos['monto'],
                'duracion_min': (t_out - pos['fecha_entrada']).total_seconds() / 60 if pos['fecha_entrada'] else None,
                'tp': pos['tp'],
                'sl': pos['sl'],
                'idx_entrada': pos['idx_entrada'],
                'retorno_usd': ganancia_usd,
                'entry_signal_30m': 1 if pos['tipo'] == 'LONG' else -1,
            })
            pos = None
            logger.info(f"Cierre forzado EOD aplicado en {simbolo}")
        except Exception as e:
            logger.warning(f"No se pudo realizar cierre EOD para {simbolo}: {e}")

    # Exporta motivos de descarte de señales
    if discards:
        try:
            disc_df = pd.DataFrame(discards)
            disc_path = BACKTEST_DIR / f"discarded_signals_{simbolo}.csv"
            disc_df.to_csv(disc_path, index=False)
            logger.info(f"Descartes de {simbolo}: {len(disc_df)} filas → {disc_path}")
        except Exception as e:
            logger.warning(f"No se pudo exportar descartes de {simbolo}: {e}")

    # Debug opcional: exportar TP/SL por trade
    try:
        dbg = pd.DataFrame(trades)
        if parametros.get('debug_signals', False) and not dbg.empty:
            dbg_cols = ['simbolo','tipo','fecha_entrada_real','entrada','tp','sl','fecha_salida','salida','razon','ganancia_usd']
            (dbg[dbg_cols]).to_csv(BACKTEST_DIR / 'tp_sl_debug.csv', index=False)
            logger.info('tp_sl_debug.csv exportado')
    except Exception as e:
        logger.warning(f'No se pudo exportar tp_sl_debug.csv: {e}')
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
    np.random.seed(42)
    logger.info(f"MAX_COMBOS activo: {MAX_COMBOS} | OPT_SYMBOLS_LIMIT: {OPT_SYMBOLS_LIMIT}")
    keys, all_cfgs = _sample_param_combos(param_grid, MAX_COMBOS, RNG_SEED)
    param_names = keys
    results     = []
    best_cfg    = None
    best_ret    = float("-inf")
    best_tuple  = (float('-inf'), float('-inf'), float('-inf'), float('-inf'))
    total_cfgs  = len(all_cfgs)

    global DETAILS
    prev_details = DETAILS
    DETAILS = False
    prev_level = logger.level
    logger.setLevel(logging.WARNING)

    for i, combo_vals in enumerate(all_cfgs):
        combo = [combo_vals[k] for k in param_names]
        cfg = base_cfg.copy()
        cfg.update(dict(zip(param_names, combo)))

        # Backtest
        pf = backtest_with_vectorbt(cfg)

        # Métricas por símbolo → promedios (multi‑criterio)
        rets, sharpes, winrates, pfs = [], [], [], []
        for sym in pf.wrapper.columns:
            stats = pf.stats(column=sym)
            rets.append(stats.loc['Total Return [%]'])
            # Algunos stats pueden faltar según versión; usa get con fallback
            sharpes.append(float(stats.get('Sharpe Ratio', np.nan)))
            winrates.append(float(stats.get('Win Rate [%]', np.nan)))
            pfs.append(float(stats.get('Profit Factor', np.nan)))
        mean_ret     = float(np.nanmean(rets))
        sharpe_mean  = float(np.nanmean(sharpes))
        winrate_mean = float(np.nanmean(winrates))
        pf_mean      = float(np.nanmean(pfs))

        row = {**dict(zip(param_names, combo)),
               'mean_total_return_%': mean_ret,
               'sharpe_mean': sharpe_mean,
               'winrate_mean_%': winrate_mean,
               'profit_factor_mean': pf_mean}
        results.append(row)

        # Selección por tupla (primario: retorno; desempates: sharpe, winrate, PF)
        sel_tuple = (mean_ret, sharpe_mean, winrate_mean, pf_mean)
        if sel_tuple > best_tuple:
            best_tuple = sel_tuple
            best_ret = mean_ret
            best_cfg = cfg.copy()

        _progress(i, total_cfgs, prefix='Optimizando grid')

    DETAILS = prev_details
    logger.setLevel(prev_level)

    # Guarda grid completo
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by=['mean_total_return_%','sharpe_mean','winrate_mean_%','profit_factor_mean'], ascending=[False, False, False, False]).reset_index(drop=True)
    top_row = df_sorted.iloc[0].to_dict() if not df_sorted.empty else None
    # Verificación de coherencia entre best_cfg y top_row
    if top_row is not None:
        mismatch = False
        for k in param_names:
            if k in top_row and best_cfg.get(k) != top_row.get(k):
                mismatch = True
                break
        if mismatch:
            logger.warning('Inconsistencia: best_cfg no coincide con la primera fila ordenada del gridsearch. Revisar criterios.')
    df_sorted.to_csv(BACKTEST_DIR / 'gridsearch_results.csv', index=False)
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
        'rsi_long': 35,
        'rsi_short': 65,
        'adx_min': 23,
        'vol_thr': 0.58,
        'tp_mult': 6,          # múltiplo de ATR para TP (más realista)
        'sl_mult': 2,
        'fee_pct': 0.0007,
        'slippage_pct': 0.0005,
        'vol_win': 50,
        'max_trades': 10,
        'stake': 'fixed',
        'stake_value': 30,
        'capital_inicial': 2000,
        'rsi_lb': 3,  # lookback de RSI para confirmar nivel en últimas N velas
        'rel_v_5m_factor': 2.0,
    }

    # ---- Quick Test overrides ----
    if QUICK_TEST:
        parametros['adx_min'] = 0       # sin filtro ADX
        parametros['vol_thr'] = 0.0     # sin filtro volumen relativo
        parametros['vol_win'] = 10      # ventana más corta de volumen
        parametros['max_trades'] = 999  # ilimitado en test rápido
        parametros['debug_signals'] = True
        parametros['rsi_long'] = 50     # más permisivo para entradas long
        parametros['rsi_short'] = 50    # igual para short
        parametros['rsi_lb'] = 10       # más margen de validación RSI
        parametros['allow_fallback'] = True  # siempre permitir fallback sin filtros
    else:
        # Valores razonables para test general
        parametros['adx_min'] = 18        # filtra mercados muy planos
        parametros['vol_thr'] = 0.25      # volumen relativo moderado
        parametros['vol_win'] = 30        # ventana de volumen más estable
        parametros['max_trades'] = 50     # límite razonable de trades
        parametros['debug_signals'] = False
        parametros['rsi_long'] = 45       # entradas long más permisivas que en producción rígida
        parametros['rsi_short'] = 55      # entradas short más permisivas que en producción rígida
        parametros['rsi_lb'] = 8          # margen RSI intermedio
        parametros['allow_fallback'] = True  # permitir fallback si no hay señales

    # --------------------------------------------------
    #  Activar grid‑search
    # --------------------------------------------------
    OPTIMIZE = not QUICK_TEST  # en quick test saltamos la búsqueda
    # Cargar best_cfg.json si no vamos a optimizar (producción/quick run)
    cfg_path = Path(__file__).resolve().parent / 'best_cfg.json'
    if not OPTIMIZE and cfg_path.exists():
        try:
            with open(cfg_path, 'r') as fp:
                best = json.load(fp)
            parametros.update(best)
            logger.info(f"Cargando parámetros desde {cfg_path}")
        except Exception as e:
            logger.warning(f"No se pudo cargar {cfg_path}: {e}")
    if OPTIMIZE:
        if FAST_OPT:
            logger.info("Usando modo rápido de optimización (grid compacto)")
            param_grid = {
                'ema_s': [10, 16, 20],
                'ema_l': [40, 50, 60],
                'adx_min': [15, 18],
                'rsi_long': [40, 45, 50],
                'rsi_short': [50, 55, 60],
                'rsi_lb': [6, 8],
                'vol_thr': [0.15, 0.25],
                'rel_v_5m_factor': [1.8, 2.0, 2.2],
            }
            if FAST_OPT and not QUICK_TEST:
                logger.warning("Modo rápido activo: los resultados pueden variar respecto al grid completo")
        else:
            # --- Grid completo actual ---
            param_grid = {
                'ema_s': [10, 16, 20, 30],
                'ema_l': [50, 100],
                'rsi': [14],
                'rsi_long': [30, 35, 40],
                'rsi_short': [60, 65, 70],
                'tp_mult': [1, 3, 5, 10],
                'sl_mult': [1, 2, 3, 5],
                'adx_min': [15, 18, 23],
                'vol_thr': [0.40, 0.50, 0.58],
                'rel_v_5m_factor': [1.5, 2.0, 2.5],
            }
        parametros['opt_symbols_limit'] = OPT_SYMBOLS_LIMIT
        best_cfg, grid_df = optimize_parameters(parametros, param_grid)
        parametros = best_cfg.copy()
        logger.info("\n==== 🏁 Primera pasada (coarse) terminada: iniciando refinamiento fine ====")
        # Segunda etapa (fine) desde el TOP N del grid rápido
        if FAST_OPT and COARSE_FINE and not QUICK_TEST:
            try:
                # Ordena por los mismos criterios usados al exportar
                sort_cols = ['mean_total_return_%','sharpe_mean','winrate_mean_%','profit_factor_mean']
                ascending = [False, False, False, False]
                if not set(sort_cols).issubset(grid_df.columns):
                    # fallback: usa retorno si faltan columnas
                    sort_cols = ['mean_total_return_%']
                    ascending = [False]
                top_df = grid_df.sort_values(by=sort_cols, ascending=ascending).head(TOP_N_REFINE)
                refine_keys = list(param_grid.keys())
                refined_grid = _refine_grid_from_top(top_df, refine_keys)
                from math import prod
                sizes = {k: len(v) for k,v in refined_grid.items()}
                est_total = prod([n for n in sizes.values()]) if sizes else 0
                logger.info(f"Grid refinado generado: tamaños={sizes}")
                logger.info(f"Tamaño final del grid refinado (combinaciones estimadas): {est_total}")
                best_cfg2, grid_df2 = optimize_parameters(parametros, refined_grid)
                parametros = best_cfg2.copy()
                logger.info("Coarse→Fine completo: parámetros refinados aplicados")
                logger.info("==== ✅ Refinamiento fine terminado: ejecutando backtest con parámetros refinados ====")
                # Guarda también el grid de la segunda pasada
                grid_df2.to_csv(BACKTEST_DIR / 'gridsearch_results_refined.csv', index=False)
            except Exception as e:
                logger.warning(f"No se pudo ejecutar el refinamiento fine: {e}")
        parametros['opt_symbols_limit'] = 0  # usa todos los símbolos para el backtest final

    # Ejecutar backtest optimizado con vectorbt
    pf = backtest_with_vectorbt(parametros)

    # --- Resumen legible por símbolo + texto humano ---
    human = resumen_humano(pf)

    # ---- Exportar resumen (CSV) ----
    summary_path = BACKTEST_DIR / "summary_per_symbol.csv"
    human.to_csv(summary_path)
    logger.info(f"Resumen por símbolo guardado en {summary_path}")

    # --- Datos para gráfico compuesto (simplificado) ---
    try:
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
    except Exception as e:
        logger.warning(f"No se pudo generar el gráfico compuesto: {e}")

    # ---- Exportar trades detallados para comparación con P&L real ----
    try:
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
        export_cols = ['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl_usd']
        trades_std = rec_df[export_cols]
        std_path = BACKTEST_DIR / "sim_trades.csv"
        trades_std.to_csv(std_path, index=False)
        logger.info(f"Trades simulados exportados a {std_path}")
    except Exception as e:
        logger.warning(f"No se pudo exportar trades simulados estándar: {e}")

    # ==================================================
    #  Simulación realista: señal 30m + ejecución en 5m
    # ==================================================
    try:
        df5_all = cargar_velas()
        if QUICK_TEST:
            # Limita a 5 símbolos para acelerar y hacer visible la tubería
            all_syms = sorted(df5_all['symbol'].unique().tolist())
            keep = all_syms[:5]
            df5_all = df5_all[df5_all['symbol'].isin(keep)]
            logger.info(f"QuickTest activos en símbolos: {keep}")
        # --- Precompute 5m indicators once (saves time across tp/sl loops) ---
        try:
            df5_all = df5_all.sort_values(['symbol','date']).copy()
            atr_win_5m = parametros.get('atr', 14)
            ema_s_win_5m = parametros.get('ema_s', 16)
            ema_l_win_5m = parametros.get('ema_l', 50)
            if 'ATR_5m' not in df5_all.columns:
                tr1 = df5_all['high'] - df5_all['low']
                tr2 = (df5_all['high'] - df5_all['close'].shift()).abs()
                tr3 = (df5_all['low'] - df5_all['close'].shift()).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df5_all['ATR_5m'] = tr.groupby(df5_all['symbol']).apply(lambda s: s.ewm(alpha=1/atr_win_5m, adjust=False).mean()).reset_index(level=0, drop=True)
            if 'EMA_S_5m' not in df5_all.columns:
                df5_all['EMA_S_5m'] = df5_all.groupby('symbol')['close'].transform(lambda s: s.ewm(span=ema_s_win_5m, adjust=False).mean())
            if 'EMA_L_5m' not in df5_all.columns:
                df5_all['EMA_L_5m'] = df5_all.groupby('symbol')['close'].transform(lambda s: s.ewm(span=ema_l_win_5m, adjust=False).mean())
            logger.info("Indicadores 5m precomputados (ATR_5m, EMA_S_5m, EMA_L_5m)")
        except Exception as e:
            logger.warning(f"No se pudo precomputar indicadores 5m: {e}")
        if df5_all.empty:
            logger.warning("No hay datos de 5m para simulación realista.")
        else:
            symbols = sorted(df5_all['symbol'].unique().tolist())
            # Capital total dividido entre símbolos en evaluación
            capital_total = float(parametros.get('capital_inicial', 300))
            if len(symbols) > 0:
                capital_por_simbolo = capital_total / len(symbols)
            else:
                capital_por_simbolo = capital_total
            logger.info(f"Capital por símbolo: {capital_por_simbolo:.2f} USD (total {capital_total} / {len(symbols)})")

            # --- Precompute 30m indicators + signals once per symbol (independent of tp/sl) ---
            df30_sig_map = {}
            for sym in symbols:
                df5_sym = df5_all[df5_all['symbol'] == sym].copy()
                if df5_sym.empty:
                    continue
                df30_sym = resample_5m_to_30m(df5_sym)
                if df30_sym.empty:
                    continue
                df30_ind = calcular_indicadores(df30_sym, parametros)
                if df30_ind.empty:
                    continue
                df30_sig = generar_senales(df30_ind, parametros)
                df30_sig_map[sym] = df30_sig

            # Candidatos de TP/SL para selección (rápida en QuickTest)
            tp_sl_candidates = parametros.get('tp_sl_candidates', [(3,2),(5,2),(6,2),(8,3)])
            best_pair = None
            best_total = float('-inf')
            best_trades_df = None

            # Búsqueda del mejor par TP/SL por retorno total USD
            for tp_mult, sl_mult in tp_sl_candidates:
                trades_realista_list = []
                for sym in symbols:
                    df30_sig = df30_sig_map.get(sym)
                    if df30_sig is None or df30_sig.empty:
                        continue
                    # Diagnóstico de cantidad de señales 30m (independientes de tp/sl)
                    try:
                        sig_count = int((df30_sig['signal'] != 0).sum())
                    except Exception:
                        sig_count = 0
                    logger.info(f"{sym}: señales 30m = {sig_count} (tp={tp_mult}, sl={sl_mult})")
                    trades_sym = simular_trades_realista(
                        df_30m=df30_sig,
                        df_5m=df5_all,
                        parametros={**parametros, 'tp_mult': tp_mult, 'sl_mult': sl_mult},
                        simbolo=sym,
                        capital_inicial=capital_por_simbolo,
                        max_trades=parametros.get('max_trades', 10)
                    )
                    if not trades_sym.empty:
                        trades_realista_list.append(trades_sym)
                if trades_realista_list:
                    candidate_trades = pd.concat(trades_realista_list, ignore_index=True)
                    resumen_cand = analizar_resultados(candidate_trades)
                    logger.info(f"Candidato tp={tp_mult}, sl={sl_mult} → trades={resumen_cand['total_trades']} PnL={resumen_cand['retorno_total_usd']:.2f}")
                    if resumen_cand['retorno_total_usd'] > best_total:
                        best_total = resumen_cand['retorno_total_usd']
                        best_pair = (tp_mult, sl_mult)
                        best_trades_df = candidate_trades
            if best_trades_df is not None:
                best_tp, best_sl = best_pair
                realista_path = BACKTEST_DIR / f"sim_trades_realista_best_tp{best_tp}_sl{best_sl}.csv"
                best_trades_df.to_csv(realista_path, index=False)
                resumen = analizar_resultados(best_trades_df)
                logger.info(
                    "Realista 30m→5m BEST | tp=%d sl=%d | trades=%d | winrate=%.1f%% | PF=%.2f | PnL=%.2f USD",
                    best_tp, best_sl, resumen['total_trades'], resumen['winrate'], resumen['profit_factor'], resumen['retorno_total_usd']
                )
                # --- Actualiza best_cfg.json con el mejor TP/SL realista ---
                try:
                    best_cfg_path = BACKTEST_DIR / "best_cfg.json"
                    cfg = {}
                    if best_cfg_path.exists():
                        with open(best_cfg_path, "r", encoding="utf-8") as f:
                            try:
                                cfg = json.load(f) or {}
                            except json.JSONDecodeError:
                                cfg = {}

                    # Mantén el resto de parámetros y pisa solo TP/SL con los ganadores del realista
                    cfg["tp"] = int(best_tp)
                    cfg["sl"] = int(best_sl)
                    # Asegura que el factor de confirmación 5m quede persistido para el bot
                    if 'rel_v_5m_factor' not in cfg:
                        cfg['rel_v_5m_factor'] = parametros.get('rel_v_5m_factor', 2.0)

                    with open(best_cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg, f, ensure_ascii=False, indent=2)

                    logger.info(f"best_cfg.json actualizado con TP={cfg['tp']}, SL={cfg['sl']}, rel_v_5m_factor={cfg.get('rel_v_5m_factor')}")
                except Exception as e:
                    logger.warning(f"No se pudo actualizar best_cfg.json con TP/SL: {e}")
                logger.info(f"Simulación realista (mejor TP/SL) exportada en {realista_path}")

                # ---- Un solo archivo de señales (ALL) para el mejor TP/SL ----
                try:
                    all_signals_rows = []
                    cols_dbg = ['date','open','high','low','close','volume','EMA_S','EMA_L','RSI','ADX','ATR','Rel_Volume','signal']
                    for sym in symbols:
                        df30_sig_b = df30_sig_map.get(sym)
                        if df30_sig_b is None or df30_sig_b.empty:
                            continue
                        sig_rows = df30_sig_b.loc[df30_sig_b['signal'] != 0, cols_dbg].copy()
                        if not sig_rows.empty:
                            sig_rows['symbol'] = sym
                            all_signals_rows.append(sig_rows[['symbol'] + cols_dbg])
                    if all_signals_rows:
                        signals_all = pd.concat(all_signals_rows, ignore_index=True)
                        sig_all_path = BACKTEST_DIR / 'signals_30m_all.csv'
                        signals_all.to_csv(sig_all_path, index=False)
                        logger.info(f"Señales 30m combinadas exportadas en {sig_all_path}")
                    else:
                        logger.info("No hubo señales 30m para exportar en el conjunto ALL.")
                except Exception as e:
                    logger.warning(f"No se pudo generar signals_30m_all.csv: {e}")

                # ---- Resumen compacto y legible por símbolo (TOP 5) ----
                try:
                    dfb = best_trades_df.copy()
                    # KPIs por símbolo
                    grp = dfb.groupby('simbolo')
                    kpis = grp['ganancia_usd'].agg(['count','sum'])
                    kpis = kpis.rename(columns={'count':'trades','sum':'pnl_usd'})
                    wins = dfb[dfb['ganancia_usd'] > 0].groupby('simbolo')['ganancia_usd'].sum()
                    losses = -dfb[dfb['ganancia_usd'] < 0].groupby('simbolo')['ganancia_usd'].sum()
                    kpis['winrate_%'] = grp.apply(lambda g: (g['ganancia_usd'] > 0).mean() * 100)
                    kpis['pf'] = wins.reindex(kpis.index).fillna(0) / losses.reindex(kpis.index).replace(0, np.nan)
                    kpis = kpis.fillna({'pf': np.inf})
                    top5 = kpis.sort_values('pnl_usd', ascending=False).head(5)

                    # Exportar KPIs por símbolo (todas las monedas)
                    kpis_out = kpis.reset_index().rename(columns={'simbolo':'symbol'})
                    kpis_out_path = BACKTEST_DIR / 'realista_kpis_per_symbol.csv'
                    kpis_out.to_csv(kpis_out_path, index=False)
                    logger.info(f"KPIs por símbolo (realista BEST) guardados en {kpis_out_path}")

                    # Texto bonito
                    lines = []
                    lines.append("\n🟢 Realista BEST resumen (por símbolo)")
                    lines.append(f"TP={best_tp} · SL={best_sl} · Trades totales={len(dfb)} · PnL total={dfb['ganancia_usd'].sum():.2f} USD")
                    lines.append("Top 5 por PnL:")
                    for sym, row in top5.iterrows():
                        lines.append(f" - {sym}: PnL={row['pnl_usd']:.2f} USD | trades={int(row['trades'])} | winrate={row['winrate_%']:.1f}% | PF={(row['pf'] if np.isfinite(row['pf']) else float('inf')):.2f}")
                    text = "\n".join(lines)
                    print(text)
                    # Guardar a TXT
                    out_path = BACKTEST_DIR / 'resumen_realista_best.txt'
                    with open(out_path, 'w') as f:
                        f.write(text + "\n")
                    logger.info(f"Resumen realista (BEST) guardado en {out_path}")
                except Exception as e:
                    logger.warning(f"No se pudo generar resumen realista BEST: {e}")
            else:
                logger.warning("La simulación realista no generó trades con ningún TP/SL candidato.")
    except Exception as e:
        logger.exception(f"Error en simulación realista 30m→5m: {e}")


if __name__ == "__main__":
    main()