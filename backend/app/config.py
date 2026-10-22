"""
Application configuration.

All secrets and environment-specific values (DB credentials, JWT secret,
allowed CORS origins) are read from environment variables / a .env file —
nothing sensitive is hardcoded here. See `.env.example` for the full list.
"""
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- App ----
    app_name: str = "Face Shape AI API"
    environment: str = "development"
    debug: bool = True

    # ---- Database ----
    # Example: postgresql+asyncpg://faceshape:faceshape@localhost:5432/faceshape
    database_url: str

    # ---- Auth / JWT ----
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours

    # ---- CORS ----
    # Comma-separated list of allowed origins, e.g. "http://localhost:5173,http://127.0.0.1:5173"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("jwt_secret_key")
    @classmethod
    def _secret_must_not_be_placeholder(cls, v: str) -> str:
        if not v or v.strip().lower() in {"changeme", "secret", ""}:
            raise ValueError(
                "JWT_SECRET_KEY must be set to a real random value. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return v

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — environment is only read once per process."""
    return Settings()
