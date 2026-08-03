"""
Application Configuration Module using Pydantic Settings v2.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Custom GPT Platform"
    ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8140
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Security
    SECRET_KEY: str = "super-secret-jwt-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Model Parameters
    DEVICE: str = "cpu"
    MODEL_PATH: str = "app/models/gpt_model.pt"
    VOCAB_PATH: str = "app/models/vocab.json"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def get_absolute_model_path(self) -> str:
        if os.path.isabs(self.MODEL_PATH):
            return self.MODEL_PATH
        return os.path.join(self.BASE_DIR, self.MODEL_PATH)

    def get_absolute_vocab_path(self) -> str:
        if os.path.isabs(self.VOCAB_PATH):
            return self.VOCAB_PATH
        return os.path.join(self.BASE_DIR, self.VOCAB_PATH)


settings = Settings()
