import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from databases import Database
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from config import config
import logging
from logging_config import setup_logging

setup_logging()

class DBConfig():
    """
    Database configuration class for managing SQLAlchemy and Databases connections.
    """
    def __init__(self, db_url: str, force_rollback: bool) -> None:
        """
        Initialize the DBConfig.
        Args:
            db_url (str): Database URL.
            force_rollback (bool): Whether to force rollback transactions (for testing).
        """
        self.db_url = db_url
        self.force_rollback = force_rollback
        self.logger = logging.getLogger(f"DBConfig:{db_url}")
        self._engine: AsyncEngine | None = None
        self._database: Database | None = None

    def get_engine(self) -> AsyncEngine:
        """
        Get an async SQLAlchemy engine.
        Returns:
            AsyncEngine: The SQLAlchemy async engine.
        """
        if not self._engine:
            self.logger.debug("Creating async SQLAlchemy engine.")
            self._engine = create_async_engine(
                self.db_url,
                echo=True,
            )
        return self._engine

    def get_session(self):
        """
        Get an async session maker for SQLAlchemy.
        Returns:
            async_sessionmaker: The session maker.
        """
        self.logger.debug("Creating async sessionmaker.")
        return async_sessionmaker(
            bind=self.get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def get_database(self) -> Database:
        """
        Get a Databases Database instance.
        Returns:
            Database: The database instance.
        """
        if not self._database:
            self.logger.debug("Creating Databases Database instance.")
            self._database = Database(self.db_url, force_rollback=self.force_rollback)
        return self._database
    
metadata = MetaData()
primary_database = DBConfig(config.POSTGRES_URL_PRIMARY, config.DB_FORCE_ROLL_BACK)
secondary_database = DBConfig(config.POSTGRES_URL_SECONDARY, config.DB_FORCE_ROLL_BACK)


async def initialize_databases():
    """
    Initialize (connect) both primary and secondary databases.
    """
    logger = logging.getLogger("api.database.database_config")
    try:
        logger.info("Connecting to primary database...")
        await primary_database.get_database().connect()
        logger.info("Primary database connected.")
        logger.info("Connecting to secondary database...")
        await secondary_database.get_database().connect()
        logger.info("Secondary database connected.")
    except Exception as e:
        logger.error(f"Error initializing databases")
        raise


async def close_databases():
    """
    Close (disconnect) both primary and secondary databases.
    """
    logger = logging.getLogger("api.database.database_config")
    try:
        logger.info("Disconnecting from primary database...")
        await primary_database.get_database().disconnect()
        logger.info("Primary database disconnected.")
        logger.info("Disconnecting from secondary database...")
        await secondary_database.get_database().disconnect()
        logger.info("Secondary database disconnected.")
    except Exception as e:
        logger.error(f"Error closing databases")
        raise

