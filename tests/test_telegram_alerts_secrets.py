from __future__ import annotations

import json


def test_telegram_alerter_auto_enabled_with_env_secrets(tmp_path, monkeypatch):
    import pkg.telegram_alerts as ta

    monkeypatch.setenv("TROBOT_TELEGRAM_BOT_TOKEN", "token_test")
    monkeypatch.setenv("TROBOT_TELEGRAM_CHAT_ID", "123456")
    cfg_path = tmp_path / "missing_config.json"

    alerter = ta.TelegramAlerter(config_path=cfg_path, repo_root=tmp_path)
    assert alerter.enabled is True


def test_telegram_alerter_respects_explicit_disabled_flag(tmp_path, monkeypatch):
    import pkg.telegram_alerts as ta

    monkeypatch.setenv("TROBOT_TELEGRAM_BOT_TOKEN", "token_test")
    monkeypatch.setenv("TROBOT_TELEGRAM_CHAT_ID", "123456")
    cfg_path = tmp_path / "monitor_config.json"
    cfg_path.write_text(json.dumps({"telegram": {"enabled": False}}), encoding="utf-8")

    alerter = ta.TelegramAlerter(config_path=cfg_path, repo_root=tmp_path)
    assert alerter.enabled is False
