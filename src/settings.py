from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: str = "development"
    random_seed: int = 42

    # Raw datasets
    paysim_csv: Path = Path("data/raw/paysim/PS_20174392719_1491204439457_log.csv")
    banksim_csv: Path = Path("data/raw/banksim")
    ibm_hi_csv: Path = Path("data/raw/ibm_aml/HI-Small_Trans.csv")
    ibm_li_csv: Path = Path("data/raw/ibm_aml/LI-Small_Trans.csv")

    # Standardised data products
    processed_data: Path = Path("data/processed/upi_transaction_training.parquet")
    aml_graph_data: Path = Path("data/processed/aml_graph_edges.parquet")
    recipient_archetype_data: Path = Path("data/processed/recipient_archetype_training.parquet")

    # Laptop-friendly preparation defaults. Use 0 or a very high number only after
    # the complete prototype works on the smaller samples.
    paysim_rows: int = 30_000
    banksim_rows: int = 30_000
    ibm_negative_ratio: int = 10
    ibm_max_negatives_per_file: int = 20_000

    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_group_id: str = "upi-fraud-prototype"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_dir: Path = Path("models")
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.6-luna"
    mlflow_tracking_uri: str = "file:./mlruns"
    use_synthetic_if_no_paysim: bool = True
    demo_rows: int = 30_000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data.parent.mkdir(parents=True, exist_ok=True)
    settings.aml_graph_data.parent.mkdir(parents=True, exist_ok=True)
    return settings
