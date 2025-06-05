from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from functools import lru_cache


class BaseConfig(BaseSettings):
    ENV_STATE: str
    API_KEY: str
    BASE_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DevConfig(BaseConfig):
    pass

class TestConfig(BaseConfig):
    pass

class ProdConfig(BaseConfig):
    pass

@lru_cache
def get_config() -> BaseConfig:
    env_state: str = os.getenv("ENV_STATE", "dev").lower()

    config_classes = {
        "dev": DevConfig,
        "test": TestConfig,
        "prod": ProdConfig
    }
    return config_classes[env_state]()

config: BaseConfig = get_config()