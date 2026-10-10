#!/usr/bin/env python3
"""Funding histórico y velas diarias del perpetuo en BingX (API pública), para pares que tienen
contado en BingX y en Binance. Contado de referencia = Binance (BingX sólo da ~380 días).
Salida: archivos/funding_bingx/<PAR>/{funding,perp_1d}.csv; el contado se reutiliza de archivos/funding/."""
import json, sys, time
from pathlib import Path
import pandas as pd, requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "archivos/funding_bingx"; BIN = ROOT / "archivos/funding"
B = "https://open-api.bingx.com/openApi"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
s = requests.Session()

def get(path, **p):
    for i in range(6):
        r = s.get(f"{B}/{path}", params=p, timeout=25)
        if r.status_code == 200 and r.json().get("code") == 0: return r.json().get("data")
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"{path} {p} {r.text[:120]}")

spot_bx = {x["symbol"] for x in get("spot/v1/common/symbols")["symbols"] if x.get("status") == 1}
tk = sorted(get("swap/v2/quote/ticker"), key=lambda t: -float(t["quoteVolume"]))
uni = []
for t in tk:
    sym = t["symbol"]; b = sym.replace("-", "")
    if sym in spot_bx and (BIN / b / "spot_1d.csv").exists(): uni.append(sym)
    if len(uni) >= N: break
print(len(uni), "pares con perp+contado BingX y contado Binance")

def funding(sym):
    rows, kw = [], {}
    while True:
        b = get("swap/v2/quote/fundingRate", symbol=sym, limit=1000, **kw)
        if not b: break
        rows += b; kw = {"endTime": min(x["fundingTime"] for x in b) - 1}
    if not rows: raise RuntimeError("sin funding")
    return pd.DataFrame(rows)[["fundingTime", "fundingRate"]].astype({"fundingRate": float}).drop_duplicates().sort_values("fundingTime")

def klines(sym):
    rows, end = [], int(time.time() * 1000)
    while True:
        b = get("swap/v3/quote/klines", symbol=sym, interval="1d", limit=1440, endTime=end)
        if not b: break
        rows += b; end = min(x["time"] for x in b) - 1
        if len(b) < 1440: break
    d = pd.DataFrame(rows)[["time", "close", "volume"]].astype(float).drop_duplicates("time").sort_values("time")
    d["quote_vol"] = d.close * d.volume
    return d.rename(columns={"time": "t"})[["t", "close", "quote_vol"]]

for i, sym in enumerate(uni):
    p = OUT / sym.replace("-", "")
    if (p / "funding.csv").exists(): continue
    p.mkdir(exist_ok=True)
    try:
        k, f = klines(sym), funding(sym)
    except Exception as e:
        print("falla", sym, e); continue
    k.to_csv(p / "perp_1d.csv", index=False); f.to_csv(p / "funding.csv", index=False)
    print(i, sym, len(f), pd.to_datetime(f.fundingTime.iloc[0], unit="ms").date(), flush=True)
json.dump(uni, open(OUT / "universo.json", "w"))
