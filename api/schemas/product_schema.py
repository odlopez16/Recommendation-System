import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import Table, Column, text
from sqlalchemy.dialects.postgresql import UUID, VARCHAR, FLOAT, TEXT
from api.database.database_config import metadata
import uuid


products_table = Table(
    'products_table',
    metadata,
    Column('id', UUID, primary_key=True, server_default=text('gen_random_uuid()')),
    Column('name', VARCHAR(255), nullable=False, unique=True),
    Column('price', FLOAT, nullable=False),
    Column('description', TEXT, nullable=True)
)