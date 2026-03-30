"""Carga centralizada de credenciales runtime.

Prioridad de fuentes:
1) Variables de entorno explicitas:
   - TROBOT_BINGX_APIKEY / BINGX_APIKEY
   - TROBOT_BINGX_SECRETKEY / BINGX_SECRETKEY
   - TROBOT_TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_TOKEN
   - TROBOT_TELEGRAM_CHAT_ID / TELEGRAM_CHAT_ID
2) AWS Secrets Manager (si TROBOT_AWS_SECRET_ID esta definido):
   - region: TROBOT_AWS_REGION (default us-east-1)
   - claves esperadas: BINGX_APIKEY, BINGX_SECRETKEY,
                       TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
3) Fallback vacio (no hardcoded secrets).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def _env(*names: str) -> str:
    for name in names:
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""


def _load_aws_secret(secret_id: str, region_name: str) -> Dict[str, Any]:
    if not secret_id:
        return {}
    try:
        import boto3  # type: ignore
    except Exception:
        return {}

    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        resp = client.get_secret_value(SecretId=secret_id)
        secret_string = resp.get("SecretString", "")
        if not secret_string:
            return {}
        data = json.loads(secret_string)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_AWS_SECRET_ID = _env("TROBOT_AWS_SECRET_ID")
_AWS_REGION = _env("TROBOT_AWS_REGION") or "us-east-1"
_AWS_SECRET_DATA = _load_aws_secret(_AWS_SECRET_ID, _AWS_REGION)


def _secret_or_env(secret_key: str, *env_names: str) -> str:
    v = _env(*env_names)
    if v:
        return v
    raw = _AWS_SECRET_DATA.get(secret_key, "")
    return str(raw).strip() if raw is not None else ""


# Telegram
token = _secret_or_env("TELEGRAM_BOT_TOKEN", "TROBOT_TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
chatID = _secret_or_env("TELEGRAM_CHAT_ID", "TROBOT_TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")
chat_id = chatID  # alias compat
send = (
    "https://api.telegram.org/bot" + token + "/sendMessage?chat_id=" + chatID + "&parse_mode=Markdown&text="
    if token and chatID
    else ""
)

# BingX
APIKEY = _secret_or_env("BINGX_APIKEY", "TROBOT_BINGX_APIKEY", "BINGX_APIKEY")
SECRETKEY = _secret_or_env("BINGX_SECRETKEY", "TROBOT_BINGX_SECRETKEY", "BINGX_SECRETKEY")
