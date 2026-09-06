"""Estado runtime de etapas TP/BE para orquestacion segura (Patch 5A).

No ejecuta ordenes. Solo mantiene estado persistente por (symbol, position_side)
para coordinar proteccion TP/SL y break-even.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
TP_STAGE_STATE_CSV = REPO_ROOT / "archivos" / "tp_stage_state.csv"

_PERSIST_STATUS: Dict[str, object] = {
    "ok": True,
    "error": "",
    "ts_utc": "",
}

TP_STAGE_VALUES = (
    "none",
    "tp1_live",
    "tp1_filled",
    "tp2_live",
    "tp2_filled",
    "tp3_live",
    "tp3_filled",
)
BREAK_EVEN_VALUES = ("inactive", "pending", "active")

TP_STAGE_COLUMNS = [
    "symbol",
    "position_side",
    "tp_mode",
    "tp_stage",
    "break_even_state",
    "tp_fill_confirmation_mode",
    "tp1_order_id",
    "tp2_order_id",
    "tp3_order_id",
    "tp1_qty",
    "tp2_qty",
    "tp3_qty",
    "tp1_price",
    "tp2_price",
    "tp3_price",
    "tp1_submit_position_qty",
    "tp2_submit_position_qty",
    "tp3_submit_position_qty",
    "sl_guard_until_utc",
    "stage_since_utc",
    "protective_stop",
    "updated_at_utc",
]


def _emit_state_event(category: str, severity: str = "WARN", **fields) -> None:
    """Telemetría de transiciones del estado de TP. Nunca debe romper la persistencia.

    Se agregó el 06/09/2026 al no poder reconstruir cómo la fila de BCH-USDT llegó a
    `tp_stage=none` con TPs vivos: no había ningún registro de creación, borrado ni
    cambio de `tp_mode`, así que la ventana era inobservable después del hecho.
    """
    try:
        from .lifecycle_events import emit_lifecycle_event
        emit_lifecycle_event(category, severity, **fields)
    except Exception:
        pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso_utc() -> str:
    return _now_utc().isoformat().replace("+00:00", "Z")


def _set_persist_status(ok: bool, error: str = "") -> None:
    _PERSIST_STATUS["ok"] = bool(ok)
    _PERSIST_STATUS["error"] = str(error or "")
    _PERSIST_STATUS["ts_utc"] = _now_iso_utc()


def get_tp_state_persist_status() -> Dict[str, object]:
    return dict(_PERSIST_STATUS)


def is_tp_state_persist_healthy() -> bool:
    return bool(_PERSIST_STATUS.get("ok", True))


def _norm_symbol(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def _norm_side(position_side: object) -> str:
    side = str(position_side or "").strip().upper()
    if side not in ("LONG", "SHORT"):
        return ""
    return side


def _safe_float_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    txt = str(value).strip()
    if txt == "" or txt.lower() in ("none", "nan", "na"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _load_state_df() -> pd.DataFrame:
    if not TP_STAGE_STATE_CSV.exists():
        return pd.DataFrame(columns=TP_STAGE_COLUMNS)
    try:
        df = pd.read_csv(TP_STAGE_STATE_CSV)
    except Exception:
        return pd.DataFrame(columns=TP_STAGE_COLUMNS)
    for c in TP_STAGE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["position_side"] = df["position_side"].astype(str).str.upper().str.strip()
    return df[TP_STAGE_COLUMNS].copy()


def _save_state_df(df: pd.DataFrame) -> bool:
    TP_STAGE_STATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in TP_STAGE_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    out = out[TP_STAGE_COLUMNS]
    tmp_path: Optional[Path] = None
    try:
        fd, tmp_raw = tempfile.mkstemp(
            prefix=f".{TP_STAGE_STATE_CSV.stem}.",
            suffix=".tmp",
            dir=str(TP_STAGE_STATE_CSV.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_raw)
        out.to_csv(tmp_path, index=False)
        os.replace(str(tmp_path), str(TP_STAGE_STATE_CSV))
        _set_persist_status(True, "")
        return True
    except Exception as exc:
        _set_persist_status(False, str(exc))
        print(f"Error persistiendo tp_stage_state.csv: {exc}")
        if tmp_path is not None:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
        return False


def _default_row(symbol: str, position_side: str) -> Dict:
    return {
        "symbol": symbol,
        "position_side": position_side,
        "tp_mode": "legacy_market_tp",
        "tp_stage": "none",
        "break_even_state": "inactive",
        "tp_fill_confirmation_mode": "inferred",
        "tp1_order_id": "",
        "tp2_order_id": "",
        "tp3_order_id": "",
        "tp1_qty": None,
        "tp2_qty": None,
        "tp3_qty": None,
        "tp1_price": None,
        "tp2_price": None,
        "tp3_price": None,
        "tp1_submit_position_qty": None,
        "tp2_submit_position_qty": None,
        "tp3_submit_position_qty": None,
        "sl_guard_until_utc": "",
        "stage_since_utc": _now_iso_utc(),
        "protective_stop": None,
        "updated_at_utc": _now_iso_utc(),
    }


def get_tp_state(symbol: object, position_side: object) -> Dict:
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return _default_row(sym, side)
    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if not m.any():
        return _default_row(sym, side)
    row = df[m].tail(1).iloc[0].to_dict()
    row["symbol"] = sym
    row["position_side"] = side
    return row


def upsert_tp_state(
    symbol: object,
    position_side: object,
    *,
    tp_mode: Optional[str] = None,
    tp_stage: Optional[str] = None,
    break_even_state: Optional[str] = None,
    tp_fill_confirmation_mode: Optional[str] = None,
    tp1_order_id: Optional[str] = None,
    tp2_order_id: Optional[str] = None,
    tp3_order_id: Optional[str] = None,
    tp1_qty=None,
    tp2_qty=None,
    tp3_qty=None,
    tp1_price=None,
    tp2_price=None,
    tp3_price=None,
    tp1_submit_position_qty=None,
    tp2_submit_position_qty=None,
    tp3_submit_position_qty=None,
    sl_guard_until_utc: Optional[str] = None,
    protective_stop=None,
) -> Dict:
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return _default_row(sym, side)

    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if m.any():
        row = df[m].tail(1).iloc[0].to_dict()
        df = df[~m].copy()
    else:
        row = _default_row(sym, side)
        # Una fila creada desde defaults por un upsert que NO declara la etapa es
        # anómala: ese llamador (set_break_even_state, bump_protective_stop...) asume
        # que la fila ya existe. Si aparece, alguien borró el estado de una posición
        # viva y quedó con `tp_stage=none` — el pozo del que no se sale solo.
        if tp_stage is None:
            _emit_state_event(
                "tp_state_row_recreated",
                "CRITICAL",
                symbol=sym,
                position_side=side,
                reason="upsert_incidental_sin_tp_stage",
                detail="fila inexistente materializada con defaults (tp_stage=none, "
                       "tp_mode=legacy_market_tp); el ladder de TP queda sin seguimiento",
            )

    row["symbol"] = sym
    row["position_side"] = side
    _tp_mode_prev = str(row.get("tp_mode", "") or "").strip()
    if tp_mode is not None:
        row["tp_mode"] = str(tp_mode).strip().lower()
        # Un cambio de modo en caliente explica que el ladder deje de seguirse: el
        # camino legacy no avanza `tp_stage`, así que el ratchet y el BE-tras-TP1
        # quedan ciegos. Se registra para poder atribuirlo, no para bloquearlo.
        if _tp_mode_prev and _tp_mode_prev != row["tp_mode"]:
            _emit_state_event(
                "tp_mode_changed",
                "WARN",
                symbol=sym,
                position_side=side,
                tp_mode_anterior=_tp_mode_prev,
                tp_mode_nuevo=row["tp_mode"],
                tp_stage=str(row.get("tp_stage", "")),
            )
    if tp_stage is not None:
        tp_stage_v = str(tp_stage).strip().lower()
        if tp_stage_v in TP_STAGE_VALUES:
            # `stage_since_utc` marca cuándo cambió la ETAPA, no cuándo se tocó la fila.
            # `updated_at_utc` se reescribe en CADA upsert (incluido `set_break_even_state`,
            # que corre cada ciclo), así que no sirve como reloj: el ratchet lo usaba y se
            # auto-bloqueaba 30 min cada vez que disparaba.
            if str(row.get("tp_stage", "")).strip().lower() != tp_stage_v:
                row["stage_since_utc"] = _now_iso_utc()
            row["tp_stage"] = tp_stage_v
            # Volver a "none" es el reset de plan de TPs al abrir posición: los
            # tramos de la posición anterior no deben heredarse (la base del
            # reparto es el tamaño de ESTA posición).
            if tp_stage_v == "none":
                # Reset de apertura: el stop protector es de la posición ANTERIOR.
                # Heredarlo pinaría el candado de monotonía en un precio ajeno.
                row["protective_stop"] = None
                for idx in (1, 2, 3):
                    row[f"tp{idx}_order_id"] = ""
                    row[f"tp{idx}_qty"] = None
                    row[f"tp{idx}_price"] = None
                    row[f"tp{idx}_submit_position_qty"] = None
    if break_even_state is not None:
        be_v = str(break_even_state).strip().lower()
        if be_v in BREAK_EVEN_VALUES:
            row["break_even_state"] = be_v
    if tp_fill_confirmation_mode is not None:
        row["tp_fill_confirmation_mode"] = str(tp_fill_confirmation_mode).strip().lower()

    if tp1_order_id is not None:
        row["tp1_order_id"] = str(tp1_order_id).strip()
    if tp2_order_id is not None:
        row["tp2_order_id"] = str(tp2_order_id).strip()
    if tp3_order_id is not None:
        row["tp3_order_id"] = str(tp3_order_id).strip()

    if tp1_qty is not None:
        row["tp1_qty"] = _safe_float_or_none(tp1_qty)
    if tp2_qty is not None:
        row["tp2_qty"] = _safe_float_or_none(tp2_qty)
    if tp3_qty is not None:
        row["tp3_qty"] = _safe_float_or_none(tp3_qty)
    if tp1_price is not None:
        row["tp1_price"] = _safe_float_or_none(tp1_price)
    if tp2_price is not None:
        row["tp2_price"] = _safe_float_or_none(tp2_price)
    if tp3_price is not None:
        row["tp3_price"] = _safe_float_or_none(tp3_price)
    if tp1_submit_position_qty is not None:
        row["tp1_submit_position_qty"] = _safe_float_or_none(tp1_submit_position_qty)
    if tp2_submit_position_qty is not None:
        row["tp2_submit_position_qty"] = _safe_float_or_none(tp2_submit_position_qty)
    if tp3_submit_position_qty is not None:
        row["tp3_submit_position_qty"] = _safe_float_or_none(tp3_submit_position_qty)

    if sl_guard_until_utc is not None:
        row["sl_guard_until_utc"] = str(sl_guard_until_utc).strip()

    if protective_stop is not None:
        row["protective_stop"] = _safe_float_or_none(protective_stop)

    row["updated_at_utc"] = _now_iso_utc()
    df.loc[len(df)] = {c: row.get(c, "") for c in TP_STAGE_COLUMNS}
    save_ok = _save_state_df(df)
    row["persist_ok"] = bool(save_ok)
    row["persist_error"] = str(get_tp_state_persist_status().get("error", ""))
    return row


def clear_tp_state(symbol: object, position_side: object, source: str = "") -> None:
    """Borra el estado de TP. `source` queda registrado: borrar el estado de una
    posición viva deja el ladder sin seguimiento y no hay forma de recuperarlo solo."""
    sym = _norm_symbol(symbol)
    side = _norm_side(position_side)
    if not sym or not side:
        return
    df = _load_state_df()
    m = (df["symbol"] == sym) & (df["position_side"] == side)
    if not m.any():
        return
    _prev = df[m].tail(1).iloc[0].to_dict()
    df = df[~m].copy()
    _save_state_df(df)
    _emit_state_event(
        "tp_state_cleared",
        "INFO",
        symbol=sym,
        position_side=side,
        source=str(source or "sin_origen"),
        tp_stage_previo=str(_prev.get("tp_stage", "")),
        tp_mode_previo=str(_prev.get("tp_mode", "")),
    )


def set_tp_submitted(
    symbol: object,
    position_side: object,
    tp_idx: int,
    *,
    order_id: str = "",
    qty=None,
    price=None,
    submit_position_qty=None,
    tp_mode: Optional[str] = None,
    fill_confirmation_mode: Optional[str] = None,
) -> Dict:
    idx = int(tp_idx)
    stage = "tp1_live" if idx <= 1 else ("tp2_live" if idx == 2 else "tp3_live")

    # La base del reparto es el tamaño de la posición cuando se armó el plan de
    # TPs, no la posición viva al momento de este submit. Si el mismo tramo se
    # recoloca (la orden previa desapareció por fill sin confirmar), conservar
    # la base original: degradarla encoge cada tramo sucesivo y deja un
    # remanente que ningún TP alcanza a cubrir.
    base_prev = _safe_float_or_none(
        get_tp_state(symbol, position_side).get(f"tp{max(1, min(3, idx))}_submit_position_qty")
    )
    base_new = _safe_float_or_none(submit_position_qty)
    if base_prev is not None and base_prev > 0:
        if base_new is None or base_new <= 0 or base_new < base_prev:
            submit_position_qty = base_prev

    kwargs = {
        "tp_stage": stage,
        "tp_mode": tp_mode,
        "tp_fill_confirmation_mode": fill_confirmation_mode,
    }
    if idx == 1:
        kwargs.update(
            {
                "tp1_order_id": order_id,
                "tp1_qty": qty,
                "tp1_price": price,
                "tp1_submit_position_qty": submit_position_qty,
            }
        )
    elif idx == 2:
        kwargs.update(
            {
                "tp2_order_id": order_id,
                "tp2_qty": qty,
                "tp2_price": price,
                "tp2_submit_position_qty": submit_position_qty,
            }
        )
    else:
        kwargs.update(
            {
                "tp3_order_id": order_id,
                "tp3_qty": qty,
                "tp3_price": price,
                "tp3_submit_position_qty": submit_position_qty,
            }
        )
    return upsert_tp_state(symbol, position_side, **kwargs)


def set_tp_filled(symbol: object, position_side: object, tp_idx: int) -> Dict:
    idx = int(tp_idx)
    if idx <= 1:
        return upsert_tp_state(symbol, position_side, tp_stage="tp1_filled", break_even_state="pending")
    if idx == 2:
        return upsert_tp_state(symbol, position_side, tp_stage="tp2_filled")
    return upsert_tp_state(symbol, position_side, tp_stage="tp3_filled")


def set_break_even_state(symbol: object, position_side: object, state: str) -> Dict:
    return upsert_tp_state(symbol, position_side, break_even_state=state)


def set_sl_guard(symbol: object, position_side: object, seconds: int = 20) -> Dict:
    try:
        sec = max(1, int(seconds))
    except Exception:
        sec = 20
    until = (_now_utc() + timedelta(seconds=sec)).isoformat().replace("+00:00", "Z")
    return upsert_tp_state(symbol, position_side, sl_guard_until_utc=until)


def is_sl_guard_active(symbol: object, position_side: object, now_utc: Optional[datetime] = None) -> bool:
    state = get_tp_state(symbol, position_side)
    raw = str(state.get("sl_guard_until_utc") or "").strip()
    if not raw:
        return False
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    now = now_utc if now_utc is not None else _now_utc()
    return bool(ts.to_pydatetime() > now)


def get_stage_since_utc(symbol: object, position_side: object) -> str:
    """Cuándo cambió la ETAPA de TP por última vez.

    Distinto de `updated_at_utc`, que se reescribe en cada upsert. Es el reloj que debe
    usar cualquier regla basada en "cuánto lleva esperando el tramo siguiente".
    """
    st = get_tp_state(symbol, position_side)
    v = str(st.get("stage_since_utc", "") or "").strip()
    return v or str(st.get("updated_at_utc", "") or "").strip()


def get_protective_stop(symbol: object, position_side: object):
    """Mejor stop protector alcanzado, o None."""
    return _safe_float_or_none(get_tp_state(symbol, position_side).get("protective_stop"))


def bump_protective_stop(symbol: object, position_side: object, stop, is_long: bool):
    """Registra `stop` sólo si MEJORA al guardado. Devuelve el mejor vigente.

    Es la garantía de monotonía: el stop protector nunca retrocede. Hacía falta porque
    `potencial_nuevo_sl` se recalcula desde cero en cada ciclo a partir del SL del
    indicador, y ese valor puede ser más flojo que el que ya había puesto el ratchet
    (caso BNB del 06/09: el ratchet dejó 743,25 y el indicador lo devolvió a 756,47).
    """
    nuevo = _safe_float_or_none(stop)
    if nuevo is None:
        return get_protective_stop(symbol, position_side)
    actual = get_protective_stop(symbol, position_side)
    if actual is not None:
        mejor = max(actual, nuevo) if is_long else min(actual, nuevo)
    else:
        mejor = nuevo
    if actual is None or mejor != actual:
        upsert_tp_state(symbol, position_side, protective_stop=mejor)
    return mejor
