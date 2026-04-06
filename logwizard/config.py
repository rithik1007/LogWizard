from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    # FAB / Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2025-01-01-preview"
    llm_temperature: float = 0.1

    # Splunk
    splunk_host: str = "localhost"
    splunk_port: int = 8089
    splunk_username: str = "admin"
    splunk_password: str = ""
    splunk_scheme: str = "https"
    splunk_token: str = ""
    splunk_index: str = "main"
    splunk_sourcetype: str = ""

    # Storage
    chroma_persist_dir: str = "./data/chroma_db"
    log_files_dir: str = "./data/logs"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


settings = Settings()

# ── Application Registry ──────────────────────────────────────────
# Maps friendly application names (and aliases) to their Splunk index + sourcetype.
# Add new apps here as needed.

APP_REGISTRY: dict[str, dict[str, str]] = {
    "myaccount": {
        "index": "mya-caas-prod",
        "sourcetype": "kube:container:mya-prod",
        "description": "MyAccount — customer account management portal",
    },
    "sdp": {
        "index": "sdp-caas-nonprod",
        "sourcetype": "kube:container:sdp-dev",
        "description": "SDP — service delivery platform (dev/nonprod)",
    },
    # Add more apps below — just copy the block above and change the values.
}

# Aliases → canonical name  (case-insensitive lookup, all lowered)
APP_ALIASES: dict[str, str] = {
    "myaccount": "myaccount",
    "my-account": "myaccount",
    "my account": "myaccount",
    "mya": "myaccount",
    "mya-caas": "myaccount",
    "sdp": "sdp",
    "sdp-caas": "sdp",
    "sdp-dev": "sdp",
    "service delivery platform": "sdp",
    "enterprise data portal": "sdp",
    "edp": "sdp",
}


def resolve_app(name: str) -> dict[str, str] | None:
    """Look up an application by name or alias. Returns the registry entry or None."""
    key = name.strip().lower()
    canonical = APP_ALIASES.get(key, key)
    return APP_REGISTRY.get(canonical)
