"""Gestor de jobs en subprocess para el dashboard.

Patrón:
- Un job = un subprocess detached (`nohup`) que escribe stdout/stderr a un log.
- Metadatos del job (cmd, pid, estado, tiempos) se guardan en JSON.
- Lockfile global evita ejecuciones concurrentes del mismo `kind`.

Uso básico:
    job = start_job(["python3", "scripts/evaluate_pairs.py", "--skip_sync"], kind="evaluate_pairs")
    status = get_status(job["id"])
    tail = read_log_tail(job["id"], n=200)
    stop_job(job["id"])
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = REPO_ROOT / "archivos" / "backtesting" / "jobs"
LOGS_DIR = JOBS_DIR / "logs"
LOCK_FILE = JOBS_DIR / ".lock"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Job:
    id: str
    kind: str
    cmd: List[str]
    pid: int
    log_path: str
    started_at: str
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    stopped_by_user: bool = False
    meta: Dict = field(default_factory=dict)

    def path(self) -> Path:
        return JOBS_DIR / f"{self.id}.json"

    def save(self) -> None:
        self.path().write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def _load_job(job_id: str) -> Optional[Job]:
    p = JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return Job(**data)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    """True si el proceso existe y NO es zombie.

    `os.kill(pid, 0)` devuelve True para zombies (defunct) — un subprocess
    detached que terminó pero no fue reaped por su parent queda en estado Z
    hasta que init lo recoja. Para evitar locks colgados, usamos `ps` para
    leer el estado real.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    # Verificar que no sea zombie. `ps -o state= -p PID` devuelve la columna
    # de estado: 'Z' o 'Z+' = zombie. Vacío = no existe.
    try:
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        state = out.stdout.strip()
        if not state:
            return False
        return not state.startswith("Z")
    except Exception:
        # Si ps falla, asumimos vivo (conservador)
        return True


def _refresh_job_state(job: Job) -> Job:
    """Si el proceso terminó pero el JSON aún no lo refleja, actualiza."""
    if job.finished_at is not None:
        return job
    if not _pid_alive(job.pid):
        job.finished_at = _now_iso()
        # No podemos recuperar exit_code tras el hecho; marcamos -1 si no hubo explicit stop.
        if job.exit_code is None:
            job.exit_code = -1 if job.stopped_by_user else 0
        job.save()
    return job


# ---------------------------------------------------------------------------
# Lock management
# ---------------------------------------------------------------------------

def _active_lock() -> Optional[Job]:
    """Devuelve el Job que tiene el lock, o None si libre/stale."""
    if not LOCK_FILE.exists():
        return None
    try:
        job_id = LOCK_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    job = _load_job(job_id)
    if job is None:
        LOCK_FILE.unlink(missing_ok=True)
        return None
    job = _refresh_job_state(job)
    if job.finished_at is not None:
        LOCK_FILE.unlink(missing_ok=True)
        return None
    return job


def is_locked() -> Optional[str]:
    """Devuelve el job_id activo si hay lock, None si libre."""
    job = _active_lock()
    return job.id if job else None


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

def start_job(cmd: List[str], kind: str, meta: Optional[Dict] = None,
              cwd: Optional[Path] = None) -> Job:
    """Lanza un subprocess detached. Lanza RuntimeError si hay lock activo."""
    existing = _active_lock()
    if existing is not None:
        raise RuntimeError(f"Ya hay un job en curso: {existing.id} ({existing.kind})")

    job_id = uuid.uuid4().hex[:12]
    log_path = LOGS_DIR / f"{job_id}.log"
    log_fh = open(log_path, "w", buffering=1)  # line-buffered

    # nohup-style: detach del TTY, nueva sesión, stdout/err → log
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(cwd or REPO_ROOT),
        start_new_session=True,  # equivalente a nohup + setsid
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    job = Job(
        id=job_id,
        kind=kind,
        cmd=cmd,
        pid=proc.pid,
        log_path=str(log_path),
        started_at=_now_iso(),
        meta=meta or {},
    )
    job.save()
    LOCK_FILE.write_text(job_id, encoding="utf-8")
    return job


def stop_job(job_id: str, timeout_sec: int = 5) -> bool:
    """Envía SIGTERM al grupo del proceso. Devuelve True si estaba vivo y lo paró."""
    job = _load_job(job_id)
    if job is None:
        return False
    if not _pid_alive(job.pid):
        return False
    try:
        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
    except ProcessLookupError:
        return False

    # Espera corta; si sigue vivo, SIGKILL
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if not _pid_alive(job.pid):
            break
        time.sleep(0.2)
    if _pid_alive(job.pid):
        try:
            os.killpg(os.getpgid(job.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    job.stopped_by_user = True
    job.finished_at = _now_iso()
    job.exit_code = -1
    job.save()
    if LOCK_FILE.exists() and LOCK_FILE.read_text(encoding="utf-8").strip() == job.id:
        LOCK_FILE.unlink(missing_ok=True)
    return True


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_status(job_id: str) -> Optional[Dict]:
    job = _load_job(job_id)
    if job is None:
        return None
    job = _refresh_job_state(job)
    running = job.finished_at is None and _pid_alive(job.pid)
    duration = _duration_sec(job)
    return {
        "id": job.id,
        "kind": job.kind,
        "pid": job.pid,
        "running": running,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "stopped_by_user": job.stopped_by_user,
        "duration_sec": duration,
        "log_path": job.log_path,
        "cmd": job.cmd,
        "meta": job.meta,
    }


def _duration_sec(job: Job) -> float:
    try:
        start = datetime.strptime(job.started_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        end_str = job.finished_at or _now_iso()
        end = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def read_log_tail(job_id: str, n: int = 200) -> str:
    job = _load_job(job_id)
    if job is None:
        return ""
    p = Path(job.log_path)
    if not p.exists():
        return ""
    try:
        # Lectura tail simple: últimos 200KB como mucho
        size = p.stat().st_size
        with open(p, "rb") as f:
            if size > 200_000:
                f.seek(size - 200_000)
                f.readline()  # descartar línea parcial
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return "\n".join(lines[-n:])
    except Exception as exc:
        return f"(error leyendo log: {exc})"


def list_recent_jobs(n: int = 15, kind: Optional[str] = None) -> List[Dict]:
    files = sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if kind and data.get("kind") != kind:
            continue
        out.append(data)
        if len(out) >= n:
            break
    return out
