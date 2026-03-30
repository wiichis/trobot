from __future__ import annotations

import json

import pytest


def test_partial_limit_mode_and_distribution_normalization(tmp_path, monkeypatch):
    import pkg.live_runtime_config as lrc

    cfg_path = tmp_path / "runtime.json"
    cfg_path.write_text(
        json.dumps(
            {
                "execution_tp": {
                    "tp_mode": "PARTIAL_LIMIT_TP",
                    "tp_partial_distribution": [3, 3, 4],
                    "tp_fill_confirmation_mode": "exchange_state",
                    "tp_legacy_fallback_on_error": False,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(lrc, "DEFAULT_CONFIG_PATH", cfg_path)
    lrc.reload_live_runtime_config()

    splits = lrc.get_tp_partial_distribution()
    assert lrc.get_tp_mode() == "partial_limit_tp"
    assert splits[0] == pytest.approx(0.3, rel=1e-9)
    assert splits[1] == pytest.approx(0.3, rel=1e-9)
    assert splits[2] == pytest.approx(0.4, rel=1e-9)
    assert lrc.get_tp_fill_confirmation_mode() == "exchange_state"
    assert lrc.get_tp_legacy_fallback_on_error() is False


def test_invalid_tp_mode_falls_back_to_legacy(tmp_path, monkeypatch):
    import pkg.live_runtime_config as lrc

    cfg_path = tmp_path / "runtime_bad.json"
    cfg_path.write_text(
        json.dumps(
            {
                "execution_tp": {
                    "tp_mode": "unsupported_mode",
                    "tp_partial_distribution": [-1, 0, 0],
                    "tp_fill_confirmation_mode": "bad_mode",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(lrc, "DEFAULT_CONFIG_PATH", cfg_path)
    lrc.reload_live_runtime_config()

    assert lrc.get_tp_mode() == "legacy_market_tp"
    assert lrc.get_tp_partial_distribution() == (0.33, 0.33, 0.34)
    assert lrc.get_tp_fill_confirmation_mode() == "inferred"


def test_entry_runtime_config_normalization(tmp_path, monkeypatch):
    import pkg.live_runtime_config as lrc

    cfg_path = tmp_path / "runtime_entry.json"
    cfg_path.write_text(
        json.dumps(
            {
                "execution_entry": {
                    "entry_mode": "LIMIT_POST_ONLY",
                    "entry_limit_offset_bps": "3.5",
                    "entry_time_in_force": "PostOnly",
                    "entry_post_only": True,
                    "entry_market_fallback_on_error": False,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(lrc, "DEFAULT_CONFIG_PATH", cfg_path)
    lrc.reload_live_runtime_config()

    assert lrc.get_entry_mode() == "limit_post_only"
    assert lrc.get_entry_limit_offset_bps() == pytest.approx(3.5, rel=1e-9)
    assert lrc.get_entry_time_in_force() == "PostOnly"
    assert lrc.is_entry_post_only_enabled() is True
    assert lrc.is_entry_market_fallback_on_error_enabled() is False
