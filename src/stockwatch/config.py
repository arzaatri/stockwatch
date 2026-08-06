"""Central, env-driven configuration. Every other module reads settings from here."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "stockwatch"
    postgres_password: str = "stockwatch"
    postgres_db: str = "stockwatch"
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    kafka_bootstrap_servers: str = "localhost:19092"

    price_poll_interval_seconds: int = 600
    slow_dim_poll_interval_seconds: int = 86400

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    newsapi_api_key: str = ""

    model_stale_after_days: int = 7

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
