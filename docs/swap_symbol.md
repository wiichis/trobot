# Guía rápida para `scripts/swap_symbol.py`

Este script automatiza el reemplazo de una moneda dentro del flujo:

1. Actualiza `pkg/best_prod.json` con el nuevo símbolo (y sus parámetros si los proporcionas).
2. Actualiza `pkg/symbols.json` (fuente simple para la whitelist en despliegues con `git pull`).
3. Sincroniza el `DEFAULT_SYMBOLS` de `pkg/settings.py` como respaldo.
4. Purga el histórico del símbolo saliente en los CSV principales (`archivos/cripto_price_5m*.csv`, `indicadores.csv`).
5. Opcionalmente (por defecto lo hace) dispara `price_bingx_5m` y `actualizar_long_ultimas_12h` para que el símbolo nuevo comience a acumular velas.

> **Importante:** Ejecuta el comando en el servidor donde corre el bot, con el proceso detenido mientras dura el swap. Los CSV y el `best_prod.json` son parte del estado local; si lo haces en tu máquina y solo subes el código, producción no verá el cambio.

## Pre-requisitos

- Python 3 y las dependencias del proyecto instaladas (usar el mismo entorno donde corre `main.py`).
- Acceso de escritura al repositorio y a la carpeta `archivos`.
- Detener el scheduler/servicio antes del swap para evitar escrituras simultáneas.

## Uso básico

```bash
python3 scripts/swap_symbol.py \
  --remove DOGE-USDT \
  --add SOL-USDT \
  --params-file archivos/backtesting/sol_params.json
```

Argumentos principales:

- `--remove SYMBOL`: símbolo que sale (se normaliza a `X-Y` mayúsculas y agrega `-USDT` si falta).
- `--add SYMBOL`: símbolo nuevo que entra.
- `--params-file RUTA`: JSON con el diccionario de parámetros para `best_prod.json` (opcional).
- `--params-json "{\"tp\": 0.01}"`: alternativa inline al archivo.
- `--skip-download`: evita que se llamen `price_bingx_5m` y `actualizar_long_ultimas_12h` (útil sin conexión, no recomendado en producción).
- `--dry-run`: muestra lo que haría sin modificar archivos (ideal para verificar antes del cambio real).

## Procedimiento recomendado en producción

1. Detén el proceso que corre `main.py` (ej. `screen`, `tmux`, systemd, etc.).
2. Asegúrate de tener los últimos cambios: `git pull`.
3. Ejecuta primero un “ensayo”:
   ```bash
   python3 scripts/swap_symbol.py --remove OLD --add NEW --dry-run
   ```
   Revisa el resumen (entradas removidas, filas a purgar, etc.).
4. Ejecuta el comando real (sin `--dry-run`). Incluye `--params-file` o `--params-json` si el nuevo símbolo requiere configuración específica.
5. Espera a que termine la descarga inicial (mensajes “Captura inicial: ejecutado”). Si falla, puedes reintentar con `--skip-download` y luego correr manualmente `python3 -m pkg.price_bingx_5m`.
6. Reinicia el proceso de trading (`python3 main.py` o el servicio correspondiente).

## Verificaciones posteriores

- `python3 - <<'PY'\nfrom pkg.cfg_loader import load_best_symbols\nprint(load_best_symbols())\nPY` → confirma que la whitelist incluye el símbolo nuevo.
- `rg -n "NEW-SYMBOL" archivos/cripto_price_5m.csv` → verifica que ya hay velas.
- Vigila los logs de `price_bingx_5m`/`indicadores` en las primeras ejecuciones para detectar errores tempranos.

Con este flujo, el símbolo retirado deja de generar datos inmediatamente y el nuevo comenzará a acumular histórico listo para backtesting la semana siguiente.
