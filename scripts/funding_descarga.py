#!/usr/bin/env python3
"""Descarga funding histórico y velas diarias (contado y perpetuo) de Binance, API pública.
Universo: perpetuos USDT-M con contado USDT, ordenados por volumen 24h. Salida en archivos/funding/."""
import json, sys, time
from pathlib import Path
import pandas as pd, requests

OUT = Path(__file__).resolve().parent.parent / "archivos/funding"
F, S = "https://fapi.binance.com", "https://api.binance.com"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
s = requests.Session()

def get(url, **p):
    for i in range(5):
        r = s.get(url, params=p, timeout=20)
        if r.status_code == 200: return r.json()
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} {p} {r.status_code}")

perp = {x["symbol"] for x in get(F + "/fapi/v1/exchangeInfo")["symbols"]
        if x["contractType"] == "PERPETUAL" and x["quoteAsset"] == "USDT" and x["status"] == "TRADING"}
spot = {x["symbol"] for x in get(S + "/api/v3/exchangeInfo")["symbols"]
        if x["quoteAsset"] == "USDT" and x["status"] == "TRADING"}
tk = sorted(get(F + "/fapi/v1/ticker/24hr"), key=lambda t: -float(t["quoteVolume"]))
uni = [t["symbol"] for t in tk if t["symbol"] in perp and t["symbol"] in spot][:N]
print(len(uni), "símbolos con perpetuo y contado")

def funding(sym):
    rows, t = [], 1
    while True:
        b = get(F + "/fapi/v1/fundingRate", symbol=sym, startTime=t, limit=1000)
        if not b: break
        rows += b; t = b[-1]["fundingTime"] + 1
    return pd.DataFrame(rows)[["fundingTime", "fundingRate"]].astype({"fundingRate": float}) if rows else None

def klines(base, path, sym):
    rows, t = [], 0
    while True:
        b = get(base + path, symbol=sym, interval="1d", startTime=t, limit=1000)
        if not b: break
        rows += b; t = b[-1][0] + 1
        if len(b) < 1000: break
    d = pd.DataFrame(rows).iloc[:, [0, 4, 7]]; d.columns = ["t", "close", "quote_vol"]
    return d.astype({"close": float, "quote_vol": float})

for i, sym in enumerate(uni):
    p = OUT / sym
    if (p / "funding.csv").exists(): continue
    p.mkdir(exist_ok=True)
    fu = funding(sym)
    if fu is None: continue
    klines(F, "/fapi/v1/klines", sym).to_csv(p / "perp_1d.csv", index=False)
    klines(S, "/api/v3/klines", sym).to_csv(p / "spot_1d.csv", index=False)
    fu.to_csv(p / "funding.csv", index=False)
    print(i, sym, len(fu), flush=True)
json.dump(uni, open(OUT / "universo.json", "w"))
