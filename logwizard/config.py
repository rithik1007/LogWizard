from pydantic_settings import BaseSettings


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

    # Storage
    chroma_persist_dir: str = "./data/chroma_db"
    log_files_dir: str = "./data/logs"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
