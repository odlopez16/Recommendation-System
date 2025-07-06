import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import Table, Column, text
from sqlalchemy.dialects.postgresql import UUID, VARCHAR, FLOAT, TEXT, DATE
from api.database.database_config import metadata


products_table = Table(
    'products_table',
    metadata,
    Column('id', UUID, primary_key=True, server_default=text('gen_random_uuid()')),
    Column('name', VARCHAR(255), nullable=False, unique=True),
    Column('price', FLOAT, nullable=False),
    Column('description', TEXT, nullable=True),
    Column('image', TEXT, nullable=True),
    Column('created_at', DATE, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)