import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import VARCHAR, Table, Column, Boolean, text
from sqlalchemy.dialects.postgresql import UUID
from api.database.database_config import metadata


users_table = Table(
    'users_table',
    metadata,
    Column('id', UUID(), nullable=False, server_default=text('gen_random_uuid()')),
    Column('email', VARCHAR(255), unique=True, nullable=False),
    Column('hashed_password', VARCHAR(255), nullable=False),
    Column('is_active', Boolean, default=True)
)