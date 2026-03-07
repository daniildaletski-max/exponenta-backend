import logging
from pydantic_settings import BaseSettings
from pydantic import model_validator
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    app_name: str = "Exponenta API"
    debug: bool = False
    version: str = "0.3.0"
    environment: str = "development"

    # Database
    database_url: str = "postgresql+asyncpg://expo:expo@localhost:5432/exponenta"
    redis_url: str = "redis://localhost:6379/0"

    # Auth
    jwt_secret: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 15  # access token: 15 min
    jwt_refresh_expire_days: int = 7  # refresh token: 7 days

    # Rate limiting
    rate_limit_default_rpm: int = 60
    rate_limit_ai_rpm: int = 10

    @model_validator(mode="after")
    def _warn_default_secret(self):
        if self.jwt_secret == "CHANGE-ME-IN-PRODUCTION" and self.environment != "development":
            raise ValueError("JWT_SECRET must be changed in non-development environments")
        if self.jwt_secret == "CHANGE-ME-IN-PRODUCTION":
            logger.warning("Using default JWT_SECRET — set JWT_SECRET env var for production")
        return self

    # Market data
    polygon_api_key: str = ""
    news_api_key: str = ""
    twitter_bearer_token: str = ""

    # AI providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""

    # GPU cloud providers
    modal_token_id: str = ""
    modal_token_secret: str = ""
    runpod_api_key: str = ""
    runpod_endpoint_id: str = ""

    # Payments
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Apple Push Notifications
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_bundle_id: str = "app.exponenta.ios"

    # GPU settings
    gpu_monte_carlo_default_sims: int = 100_000
    gpu_prefer_cloud: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
