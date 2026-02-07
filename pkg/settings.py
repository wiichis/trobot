# pkg/settings.py
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BEST_CFG_PATH = PKG_DIR / "best_cfg.json"   # opcional
BEST_PROD_FALLBACK = PKG_DIR / "best_prod.json"
BEST_PROD_CONSISTENT = PKG_DIR / "best_prod_consistent.json"
# Usa consistent si existe, si no, usa el fallback.
BEST_PROD_PATH = BEST_PROD_CONSISTENT if BEST_PROD_CONSISTENT.exists() else BEST_PROD_FALLBACK  # fuente de verdad para prod

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
