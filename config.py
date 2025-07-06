from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BaseConfig(BaseSettings):
    ENV_STATE: str
    API_KEY: str
    BASE_URL: str
    DB_FORCE_ROLL_BACK: bool
    OPENAI_MODEL: str
    JWT_SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_SECRET_KEY: str
    REFRESH_TOKEN_EXPIRE_MINUTES: int
    MAX_SESSIONS_PER_USER: int = 5

    @property
    def POSTGRES_URL_PRIMARY(self) -> str:
        raise NotImplementedError

    @property
    def POSTGRES_URL_SECONDARY(self) -> str:
        raise NotImplementedError

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DevConfig(BaseConfig):
    primary_url: str = Field(..., validation_alias='DEV_POSTGRES_URL_PRIMARY')
    secondary_url: str = Field(..., validation_alias='DEV_POSTGRES_URL_SECONDARY')

    @property
    def POSTGRES_URL_PRIMARY(self) -> str:
        return self.primary_url

    @property
    def POSTGRES_URL_SECONDARY(self) -> str:
        return self.secondary_url


class TestConfig(BaseConfig):
    primary_url: str = Field(..., validation_alias='TEST_POSTGRES_URL_PRIMARY')
    secondary_url: str = Field(..., validation_alias='TEST_POSTGRES_URL_SECONDARY')
    DB_FORCE_ROLL_BACK: bool = True

    @property
    def POSTGRES_URL_PRIMARY(self) -> str:
        return self.primary_url

    @property
    def POSTGRES_URL_SECONDARY(self) -> str:
        return self.secondary_url


class ProdConfig(BaseConfig):
    primary_url: str = Field(..., validation_alias='PROD_POSTGRES_URL_PRIMARY')
    secondary_url: str = Field(..., validation_alias='PROD_POSTGRES_URL_SECONDARY')

    @property
    def POSTGRES_URL_PRIMARY(self) -> str:
        return self.primary_url

    @property
    def POSTGRES_URL_SECONDARY(self) -> str:
        return self.secondary_url


@lru_cache
def get_config() -> BaseConfig:
    # Load environment variables again to ensure they're available
    load_dotenv(override=True)
    
    env_state = os.getenv("ENV_STATE", "dev").lower()
    if env_state not in ["dev", "test", "prod"]:
        env_state = "dev"  # default to dev if invalid value
        
    config_classes = {
        "dev": DevConfig,
        "test": TestConfig,
        "prod": ProdConfig
    }
    return config_classes[env_state]()

config: BaseConfig = get_config()