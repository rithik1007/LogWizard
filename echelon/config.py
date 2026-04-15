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

    # Teams Bot (optional — only needed for Teams bot integration)
    teams_bot_id: str = ""
    teams_bot_password: str = ""

    # Daily Digest Email
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    digest_from_email: str = "echelon-ai@noreply.local"
    digest_send_hour: int = 12  # Hour of day (0-23) to send the daily digest
    digest_send_minute: int = 0
    digest_subscribers_file: str = "./data/digest_subscribers.json"

    # Azure DevOps (build/pipeline integration)
    azdevops_org: str = ""           # e.g. "wk-gbs-es"
    azdevops_project: str = ""       # e.g. "GBSInfraDevOps"
    azdevops_pat: str = ""           # Personal Access Token (Build Read + Code Read)
    azdevops_folder: str = "\\STEP-CI"  # Pipeline folder scope — only show pipelines under this path

    model_config = {"env_file": str(_ENV_FILE), "env_file_encoding": "utf-8"}


settings = Settings()

# ── Application Registry ──────────────────────────────────────────
# Maps friendly application names (and aliases) to their Splunk index + sourcetype.
# `sources` lists component/source names from local logs that belong to this app.
# Add new apps here as needed.

APP_REGISTRY: dict[str, dict] = {
    "myaccount": {
        "index": "mya-caas-prod",
        "sourcetype": "kube:container:mya-prod",
        "description": "MyAccount — customer account management portal",
        "sources": ["db-pool", "auth-service", "session-manager", "user-profile", "scheduler", "api-gateway"],
        "pipelines": ["mya", "myaccount"],  # substrings to match pipeline names
    },
    "sdp": {
        "index": "sdp-caas-nonprod",
        "sourcetype": "kube:container:sdp-dev",
        "description": "STEP Data Portal — enterprise data platform",
        "sources": ["order-service", "payment-gw", "catalog-service", "notification-svc", "inventory-service"],
        "pipelines": ["step-data-portal", "edp"],  # substrings to match pipeline names
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
    "step-data-portal": "sdp",
    "step data portal": "sdp",
    "enterprise data portal": "sdp",
    "edp": "sdp",
}


def resolve_app(name: str) -> dict[str, str] | None:
    """Look up an application by name or alias. Returns the registry entry or None."""
    key = name.strip().lower()
    canonical = APP_ALIASES.get(key, key)
    return APP_REGISTRY.get(canonical)
