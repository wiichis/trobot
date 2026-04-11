# pkg/settings.py
#
# Fuente de verdad ÚNICA: pkg/best_prod.json
# Este archivo se genera desde backtesting (--export_best + --sync_best_to_pkg)
# y define tanto la whitelist de símbolos como los parámetros por par.
#
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent

# Override por variable de entorno (útil para testing local sin tocar el repo)
_env_best_prod = os.getenv("TROBOT_BEST_PROD_PATH", "").strip()
if _env_best_prod:
    BEST_PROD_PATH = Path(_env_best_prod).expanduser()
else:
    BEST_PROD_PATH = PKG_DIR / "best_prod.json"

# Defaults seguros si best_prod.json no existe aún
DEFAULT_SYMBOLS = [
    "AVAX-USDT",
    "HBAR-USDT",
]
