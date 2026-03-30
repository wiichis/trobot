# pkg/settings.py
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BEST_CFG_PATH = PKG_DIR / "best_cfg.json"   # opcional
BEST_PROD_FALLBACK = PKG_DIR / "best_prod.json"
BEST_PROD_CONSISTENT = PKG_DIR / "best_prod_consistent.json"
BEST_PROD_LIVE_BENCHMARK = PKG_DIR / "best_prod_live_benchmark.json"

# Fuente de verdad para prod:
# 1) TROBOT_BEST_PROD_PATH (override explicito),
# 2) best_prod_live_benchmark.json (freeze alineado a benchmark validado),
# 3) best_prod_consistent.json,
# 4) best_prod.json (fallback).
_env_best_prod = os.getenv("TROBOT_BEST_PROD_PATH", "").strip()
if _env_best_prod:
    BEST_PROD_PATH = Path(_env_best_prod).expanduser()
elif BEST_PROD_LIVE_BENCHMARK.exists():
    BEST_PROD_PATH = BEST_PROD_LIVE_BENCHMARK
elif BEST_PROD_CONSISTENT.exists():
    BEST_PROD_PATH = BEST_PROD_CONSISTENT
else:
    BEST_PROD_PATH = BEST_PROD_FALLBACK

# Defaults seguros si no hay configs
DEFAULT_SYMBOLS = [
    "BCH-USDT",
    "AVAX-USDT",
    "CFX-USDT",
    "DOT-USDT",
    "NEAR-USDT",
    "APT-USDT",
    "HBAR-USDT",
    "BNB-USDT",
    "TRX-USDT",
    "DOGE-USDT"
]
