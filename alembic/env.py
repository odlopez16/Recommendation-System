import os
import sys
from logging.config import fileConfig
from sqlalchemy import create_engine, pool
from alembic import context

# Añadimos la raíz del proyecto al path para poder importar la configuración de la app
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# Importamos los metadatos que contienen TODAS nuestras tablas.
from api.database.database_config import metadata
from config import config as app_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# El target_metadata es el objeto que contiene la definición de TODAS nuestras tablas.
target_metadata = metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.
    ...
    """
    # El modo offline es más complejo con ramas. Por ahora, lo mantenemos simple
    # y enfocado en la base de datos primaria para la generación de scripts.
    primary_url = app_config.POSTGRES_URL_PRIMARY.replace("postgresql+asyncpg", "postgresql")
    context.configure(
        url=primary_url,
        target_metadata=target_metadata, # Usamos el metadata completo
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    ...
    """
    primary_sync_url = app_config.POSTGRES_URL_PRIMARY.replace("postgresql+asyncpg", "postgresql")
    secondary_sync_url = app_config.POSTGRES_URL_SECONDARY.replace("postgresql+asyncpg", "postgresql")

    engines = {
        'primary': create_engine(primary_sync_url, poolclass=pool.NullPool),
        'secondary': create_engine(secondary_sync_url, poolclass=pool.NullPool)
    }

    import logging
    logger = logging.getLogger("alembic.env")
    for name, engine in engines.items():
        logger.info(f"Running migrations for '{name}' database...")
        with engine.connect() as connection:
            def include_object(object, obj_name, type_, reflected, compare_to):
                if type_ == "table":
                    if name == 'primary':
                        return obj_name == 'embeddings_table'
                    if name == 'secondary':
                        return obj_name == 'products_table'
                return True

            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                include_object=include_object,
                compare_type=True
            )

            with context.begin_transaction():
                context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
