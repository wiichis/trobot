#!/usr/bin/env python3
"""Arma la serie de velas 5m de los pares CANDIDATOS (01/10/2026).

Fuentes:
- Historia: archivos mensuales de Binance USDT-M futures (data.binance.vision), ya
  descargados en archivos/candidatos/binance/<PAR>USDT-5m-YYYY-MM.zip.
- Últimos ~45 días: el API de BingX (el exchange donde operamos), que además cubre el mes
  en curso, todavía no publicado por Binance.

En el tramo que se solapan compara las dos fuentes (paridad de close): si no coinciden,
la historia de Binance no sirve como proxy de BingX para ese par y se avisa.

Escribe archivos/candidatos/velas_candidatos.csv (formato de cripto_price_5m_long.csv) y
archivos/candidatos/reglas_contrato.json con tick y qty_step de BingX.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pkg.price_bingx_5m import _fetch_bingx_candles  # noqa: E402

DIR = REPO / "archivos" / "candidatos"
CANDIDATOS = ["UNI", "SUI", "ARB", "ENA", "WLD", "TAO", "QNT", "1000PEPE",
              "LTC", "SOL", "NEAR", "AAVE", "XRP", "ZEC", "HYPE"]
DIAS_BINGX = 45


def binance(base: str) -> pd.DataFrame:
    frames = []
    for z in sorted((DIR / "binance").glob(f"{base}USDT-5m-*.zip")):
        with zipfile.ZipFile(z) as zf:
            raw = zf.read(zf.namelist()[0]).decode()
        first = raw.split("\n", 1)[0]
        df = pd.read_csv(io.StringIO(raw), header=0 if first.startswith("open_time") else None)
        df = df.iloc[:, :6]
        df.columns = ["open_time", "open", "high", "low", "close", "volume"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    ot = pd.to_numeric(df["open_time"])
    unidad = "us" if ot.max() > 1e14 else "ms"   # Binance pasó a microsegundos en 2025
    df["date"] = pd.to_datetime(ot, unit=unidad, utc=True)
    df["symbol"] = f"{base}-USDT"
    return df[["symbol", "open", "high", "low", "close", "volume", "date"]]


def bingx(base: str) -> pd.DataFrame:
    sym = f"{base}-USDT"
    out, end = [], None
    hasta = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DIAS_BINGX)
    for _ in range(20):
        chunk = pd.DataFrame(_fetch_bingx_candles(sym, 1000, end_time_ms=end))
        if chunk.empty:
            break
        out.append(chunk)
        earliest = chunk["date"].min()
        if earliest <= hasta:
            break
        end = int((earliest - pd.Timedelta(minutes=5)).timestamp() * 1000)
        time.sleep(0.25)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out).drop_duplicates(["date"]).sort_values("date")
    return df[df["date"] < df["date"].max()]   # la última está en formación


def reglas() -> dict:
    r = requests.get("https://open-api.bingx.com/openApi/swap/v2/quote/contracts", timeout=20).json()
    out = {}
    for c in r.get("data", []):
        base = str(c.get("symbol", "")).split("-")[0]
        if base in CANDIDATOS:
            pp, qp = int(c.get("pricePrecision", 0)), int(c.get("quantityPrecision", 0))
            out[c["symbol"]] = {"price_tick": 10 ** -pp, "qty_step": 10 ** -qp,
                                "trade_min_usdt": c.get("tradeMinUSDT"), "status": c.get("status")}
    return out


def main():
    series, informe = [], {}
    for base in CANDIDATOS:
        a, b = binance(base), bingx(base)
        if b.empty:
            informe[base] = "SIN DATOS EN BINGX"
            continue
        sol = a.merge(b, on="date", suffixes=("_bn", "_bx")) if not a.empty else pd.DataFrame()
        if len(sol):
            dif = (sol["close_bn"].astype(float) / sol["close_bx"].astype(float) - 1).abs()
            informe[base] = dict(solape_velas=int(len(sol)), dif_close_mediana_bps=round(float(dif.median() * 1e4), 2),
                                 dif_close_p99_bps=round(float(dif.quantile(0.99) * 1e4), 2),
                                 binance_desde=str(a["date"].min()), bingx_desde=str(b["date"].min()))
        else:
            informe[base] = "sin solape"
        corte = b["date"].min()
        serie = pd.concat([a[a["date"] < corte], b]) if not a.empty else b
        series.append(serie)
    df = pd.concat(series).drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    out.to_csv(DIR / "velas_candidatos.csv", index=False)
    (DIR / "reglas_contrato.json").write_text(json.dumps(reglas(), indent=2), encoding="utf-8")
    (DIR / "paridad_binance_bingx.json").write_text(json.dumps(informe, indent=2), encoding="utf-8")
    print(json.dumps(informe, indent=1))
    print(df.groupby("symbol")["date"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
