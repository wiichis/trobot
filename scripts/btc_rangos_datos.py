"""Velas 1 h de BTC/USDT spot para el bot por rangos (Fase 1, 22/10/2026).

Fuentes:
- Binance spot, mensuales y diarios de data.binance.vision en `archivos/btc/binance_spot_1h/`
  (2021-01 → lo último bajado). Es la serie larga.
- API de BingX spot (`/openApi/spot/v2/market/kline`), recorrida hacia atrás con `endTime`.
  Sirve para dos cosas: medir la paridad contra Binance en el tramo que se solapan, y
  completar hasta hoy. La última vela de BingX está en formación y se descarta siempre.

Salidas en `archivos/btc_rangos/`:
- `velas_1h.csv`: date (apertura, UTC), open, high, low, close, fuente. Binance hasta su
  última vela; BingX después.
- `paridad_1h.csv`: diferencias en bps por vela en el solape.
- `reglas_spot.json`: reglas del par en BingX (mínimos, step, tick), tal cual las da el API.

Uso: python3 scripts/btc_rangos_datos.py [--dias_bingx 200]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "archivos" / "btc_rangos"
API = "https://open-api.bingx.com"
SYMBOL = "BTC-USDT"
HORA_MS = 3600_000

_spec = importlib.util.spec_from_file_location("btc_lab", REPO / "scripts" / "btc_lab_temporalidades.py")
lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab)


def binance_spot_1h() -> pd.DataFrame:
    return lab._leer_velas("archivos/btc/binance_spot_1h", "BTCUSDT-1h-*.zip")


def _get(path: str, params: dict) -> dict:
    for intento in range(5):
        try:
            r = requests.get(API + path, params=params, timeout=15)
            d = r.json()
            if d.get("code") == 0:
                return d
            print(f"  {path} code={d.get('code')} msg={d.get('msg')}")
        except Exception as exc:
            print(f"  {path} error {exc}")
        time.sleep(1 + intento)
    raise RuntimeError(f"{path} falló 5 veces")


def bingx_spot_1h(dias: int) -> pd.DataFrame:
    """Velas 1 h cerradas de BingX spot de los últimos `dias` días."""
    ahora_ms = int(time.time() * 1000)
    desde_ms = ahora_ms - dias * 24 * HORA_MS
    filas, end = [], ahora_ms
    while end > desde_ms:
        data = _get("/openApi/spot/v2/market/kline",
                    {"symbol": SYMBOL, "interval": "1h", "limit": 1000, "endTime": end})["data"]
        if not data:
            break
        filas.extend(data)
        mas_vieja = min(int(k[0]) for k in data)
        if mas_vieja >= end:
            break
        end = mas_vieja - 1
        time.sleep(0.2)
    df = pd.DataFrame([k[:5] for k in filas], columns=["ot", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["ot"].astype("int64"), unit="ms")
    df = df.drop_duplicates("date").set_index("date").sort_index()[["open", "high", "low", "close"]].astype(float)
    # Sólo velas cerradas: la que abrió hace menos de 1 h todavía está en formación.
    cierre = df.index + pd.Timedelta(hours=1)
    df = df[cierre <= pd.Timestamp(ahora_ms, unit="ms")]
    return df[df.index >= pd.Timestamp(desde_ms, unit="ms")]


def huecos(df: pd.DataFrame) -> pd.Series:
    d = df.index.to_series().diff()
    return d[d > pd.Timedelta(hours=1)]


def paridad(bn: pd.DataFrame, bx: pd.DataFrame) -> pd.DataFrame:
    comun = bn.index.intersection(bx.index)
    out = pd.DataFrame(index=comun)
    for c in ("open", "high", "low", "close"):
        out[c + "_bps"] = (bx.loc[comun, c] / bn.loc[comun, c] - 1) * 1e4
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias_bingx", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    bn = binance_spot_1h()
    print(f"Binance spot 1h: {len(bn)} velas, {bn.index[0]} → {bn.index[-1]}")
    h = huecos(bn)
    print(f"  huecos > 1 h: {len(h)}" + (f" (mayor {h.max()}, en {h.idxmax()})" if len(h) else ""))
    for t, d in h.items():
        print(f"    {t - d} → {t}  ({d})")

    bx = bingx_spot_1h(args.dias_bingx)
    print(f"BingX spot 1h: {len(bx)} velas cerradas, {bx.index[0]} → {bx.index[-1]}")
    hb = huecos(bx)
    print(f"  huecos > 1 h: {len(hb)}")

    par = paridad(bn, bx)
    par.to_csv(OUT / "paridad_1h.csv")
    print(f"Paridad en el solape ({len(par)} velas, {par.index[0]} → {par.index[-1]}), bps BingX vs Binance:")
    for c in par.columns:
        a = par[c].abs()
        print(f"  {c:10s} mediana {a.median():6.2f} · p95 {a.quantile(.95):6.2f} · p99 {a.quantile(.99):6.2f} · "
              f"máx {a.max():7.2f} · sesgo medio {par[c].mean():+6.2f}")

    cola = bx[bx.index > bn.index[-1]]
    velas = pd.concat([bn.assign(fuente="binance"), cola.assign(fuente="bingx")])
    velas.index.name = "date"
    velas.to_csv(OUT / "velas_1h.csv")
    hv = huecos(velas)
    print(f"velas_1h.csv: {len(velas)} velas ({len(cola)} de BingX), {velas.index[0]} → {velas.index[-1]}, "
          f"huecos {len(hv)}")

    reglas = _get("/openApi/spot/v1/common/symbols", {"symbol": SYMBOL, "timestamp": int(time.time() * 1000)})
    info = reglas["data"]["symbols"][0]
    info["_consultado_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (OUT / "reglas_spot.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    precio = float(velas["close"].iloc[-1])
    print(f"Reglas BingX spot {SYMBOL}: minQty {info['minQty']} (≈{info['minQty'] * precio:.2f} USDT), "
          f"minNotional {info['minNotional']}, stepSize {info['stepSize']}, tickSize {info['tickSize']}")


if __name__ == "__main__":
    main()
