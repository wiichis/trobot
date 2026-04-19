"""SCP sync helper para descargar CSV de producción.

Reutiliza el patrón de scripts/evaluate_pairs.py::_sync_from_production.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PEM_KEY = Path.home() / "Documents" / "proyectos" / "ls_keys" / "trobot4.pem"
DEFAULT_SERVER = "ubuntu@98.81.217.194"
REMOTE_BASE = "/home/ubuntu/TRobot"

# Archivos que el dashboard necesita del servidor.
SYNC_FILES: List[Tuple[str, str]] = [
    ("archivos/PnL.csv", f"{REMOTE_BASE}/archivos/PnL.csv"),
    ("archivos/execution_ledger.csv", f"{REMOTE_BASE}/archivos/execution_ledger.csv"),
    ("archivos/lifecycle_event_log.csv", f"{REMOTE_BASE}/archivos/lifecycle_event_log.csv"),
    ("archivos/ganancias.csv", f"{REMOTE_BASE}/archivos/ganancias.csv"),
    ("archivos/trade_closed_log.csv", f"{REMOTE_BASE}/archivos/trade_closed_log.csv"),
    ("archivos/indicadores.csv", f"{REMOTE_BASE}/archivos/indicadores.csv"),
    ("archivos/sl_watch.csv", f"{REMOTE_BASE}/archivos/sl_watch.csv"),
    ("archivos/tp_stage_state.csv", f"{REMOTE_BASE}/archivos/tp_stage_state.csv"),
    ("pkg/best_prod.json", f"{REMOTE_BASE}/pkg/best_prod.json"),
]


@dataclass
class SyncResult:
    file_rel: str
    ok: bool
    size_mb: float
    detail: str


def sync_from_production(
    server: str = DEFAULT_SERVER,
    pem_key: Path = DEFAULT_PEM_KEY,
    files: Iterable[Tuple[str, str]] = SYNC_FILES,
    timeout_sec: int = 60,
) -> List[SyncResult]:
    """Descarga archivos vía SCP. Continúa aunque alguno falle."""
    results: List[SyncResult] = []

    if not pem_key.exists():
        for local_rel, _ in files:
            results.append(SyncResult(local_rel, False, 0.0, f"PEM no encontrada: {pem_key}"))
        return results

    for local_rel, remote_path in files:
        local_path = REPO_ROOT / local_rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "scp", "-i", str(pem_key),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{server}:{remote_path}",
            str(local_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout_sec)
            size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 0.0
            results.append(SyncResult(local_rel, True, size_mb, "ok"))
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr.decode() if exc.stderr else str(exc))[:200]
            results.append(SyncResult(local_rel, False, 0.0, detail))
        except subprocess.TimeoutExpired:
            results.append(SyncResult(local_rel, False, 0.0, f"timeout {timeout_sec}s"))

    return results
