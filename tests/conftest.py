from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest


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
