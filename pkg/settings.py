# pkg/settings.py
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BEST_CFG_PATH  = PKG_DIR / "best_cfg.json"   # opcional
BEST_PROD_PATH = PKG_DIR / "best_prod.json"  # fuente de verdad para prod

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