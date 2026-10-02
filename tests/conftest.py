from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVOS_DEL_REPO = REPO_ROOT / "archivos"


def _foto_archivos_del_repo() -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    if not ARCHIVOS_DEL_REPO.is_dir():
        return out
    for p in ARCHIVOS_DEL_REPO.rglob("*"):
        try:
            if p.is_file():
                st = p.stat()
                out[str(p.relative_to(REPO_ROOT))] = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    return out


@pytest.fixture(scope="session", autouse=True)
def _guardia_archivos_del_repo():
    """Falla la sesión si algún test creó, modificó o borró algo en `archivos/` del repo.

    En el checkout principal esos CSV son la copia local de los logs de prod que se usan
    para analizar. Los tests escribían ahí en silencio (filas con rutas de pytest y avisos
    `tick_size_implausible` que nunca pasaron en prod). Si esto salta, la ruta nueva va
    en `_aislar_rutas_absolutas_del_repo` o el test necesita `isolated_workspace`.
    Falso positivo posible: otro proceso (dashboard, sync con prod) escribiendo a la vez.
    """
    antes = _foto_archivos_del_repo()
    yield
    despues = _foto_archivos_del_repo()
    nuevos = sorted(set(despues) - set(antes))
    borrados = sorted(set(antes) - set(despues))
    modificados = sorted(k for k in set(antes) & set(despues) if antes[k] != despues[k])
    if nuevos or borrados or modificados:
        lineas = (
            [f"  nuevo:      {k}" for k in nuevos]
            + [f"  modificado: {k}" for k in modificados]
            + [f"  borrado:    {k}" for k in borrados]
        )
        pytest.fail(
            "La suite tocó archivos/ del repo (deben quedar en tmp_path):\n" + "\n".join(lineas),
            pytrace=False,
        )


@pytest.fixture(autouse=True)
def _aislar_rutas_absolutas_del_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirige a `tmp_path/archivos/` los CSV que `pkg/` resuelve contra REPO_ROOT.

    Estas rutas son absolutas (`Path(__file__)`), así que `isolated_workspace` (que sólo
    hace chdir) no las aísla. Apuntan al mismo `archivos/` que usa `isolated_workspace`,
    así un test que lee `./archivos/x.csv` ve lo mismo que escribió el módulo.

    El dispatcher de lifecycle además manda Telegram con la config trackeada
    (`enabled: true`): se reemplaza por una deshabilitada para que un test nunca pueda
    mandar una alerta real si en el entorno hay credenciales.
    """
    import pkg.execution_ledger as el
    import pkg.lifecycle_events as le
    import pkg.price_bingx_5m as px
    import pkg.tp_stage_state as tps

    archivos = tmp_path / "archivos"
    archivos.mkdir(parents=True, exist_ok=True)

    telegram_off = tmp_path / "telegram_deshabilitado.json"
    telegram_off.write_text(json.dumps({"telegram": {"enabled": False}}), encoding="utf-8")
    monkeypatch.setattr(le, "LIFECYCLE_LOG_CSV", archivos / "lifecycle_event_log.csv")
    monkeypatch.setattr(le, "DEFAULT_TELEGRAM_CONFIG", telegram_off)
    monkeypatch.setattr(le, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(le, "_DISPATCHER", None)

    # `query_order` consulta el exchange real dentro del job de transiciones (modo
    # exchange_state). Ningún test debe llegar a la red: por defecto responde como un
    # error del exchange, y el código cae a la inferencia por posición. Los tests que
    # la necesitan la reemplazan (exchange falso o respuesta grabada).
    import pkg.bingx as bx
    monkeypatch.setattr(
        bx, "query_order",
        lambda *_a, **_k: json.dumps({"code": -1, "msg": "red bloqueada en tests", "data": {}}),
    )

    monkeypatch.setattr(el, "DEFAULT_EXECUTION_LEDGER_PATH", archivos / "execution_ledger.csv")
    monkeypatch.setattr(tps, "TP_STAGE_STATE_CSV", archivos / "tp_stage_state.csv")

    monkeypatch.setattr(px, "CSV_PATH", archivos / "cripto_price_5m.csv")
    monkeypatch.setattr(px, "CSV_30M_PATH", archivos / "cripto_price_30m.csv")
    monkeypatch.setattr(px, "LONG_CSV_PATH", archivos / "cripto_price_5m_long.csv")
    monkeypatch.setattr(px, "BENCH_CSV_PATH", archivos / "cripto_price_5m_bench.csv")
    return archivos


@pytest.fixture(autouse=True)
def _reset_live_runtime_config_cache():
    """Aísla el runtime config entre tests.

    `get_live_runtime_config` está bajo `lru_cache`, así que un test que apunta
    `DEFAULT_CONFIG_PATH` a un temp y recarga deja la config cacheada para todos los
    que siguen — y `monkeypatch` restaura el atributo pero no el caché. Eso hacía que
    los tests del gate horario pasaran aislados y fallaran dentro de la suite.
    """
    import pkg.live_runtime_config as lrc

    lrc.reload_live_runtime_config()
    yield
    lrc.reload_live_runtime_config()


@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aisla rutas relativas ./archivos usadas por runtime."""
    (tmp_path / "archivos").mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def runtime_event_spy(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[dict]]:
    """Captura emisiones lifecycle+ledger sin tocar red ni archivos productivos."""
    import pkg.monkey_bx as mb

    out: Dict[str, List[dict]] = {"lifecycle": [], "ledger": []}

    def _emit(category, severity="INFO", **fields):
        row = {"category": category, "severity": severity, **fields}
        out["lifecycle"].append(row)
        return {"sent": True, "detail": "mocked", "ts_utc": "1970-01-01T00:00:00Z"}

    def _ledger(event_type, **fields):
        row = {"event_type": event_type, **fields}
        out["ledger"].append(row)
        return row

    monkeypatch.setattr(mb, "emit_lifecycle_event", _emit)
    monkeypatch.setattr(mb, "append_execution_ledger_event", _ledger)
    monkeypatch.setattr(mb.time, "sleep", lambda *_args, **_kwargs: None)
    return out


@pytest.fixture
def temp_tp_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirige tp_stage_state.csv a un path temporal."""
    import pkg.tp_stage_state as tps

    path = tmp_path / "archivos" / "tp_stage_state.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(tps, "TP_STAGE_STATE_CSV", path)
    return path


def make_orders_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["symbol", "orderId", "type", "side", "positionSide", "price", "stopPrice", "time"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]
