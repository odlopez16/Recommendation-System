from operator import index
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import VARCHAR, Table, Column, Boolean, text, ForeignKey
from sqlalchemy.dialects.postgresql import DATE, UUID, BOOLEAN, TIMESTAMP
from api.database.database_config import metadata


users_table = Table(
    'users_table',
    metadata,
    Column('id', UUID(), nullable=False, server_default=text('gen_random_uuid()')),
    Column('email', VARCHAR(255), unique=True, nullable=False),
    Column('hashed_password', VARCHAR(255), nullable=False),
    Column('is_active', BOOLEAN, server_default=text('true')),
    Column('created_at', DATE, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)


revoked_tokens_table = Table(
    'revoked_tokens_table',
    metadata,
    Column('id', UUID, primary_key=True),
    Column('token', VARCHAR(255), unique=True, nullable=False),
    Column('revoked_at', TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP')),
    Column('expires_at', TIMESTAMP(timezone=True), nullable=False),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users_table.id', ondelete="CASCADE"), nullable=False)
)

user_sessions_table = Table(
    'user_sessions_table',
    metadata,
    Column('id', UUID, primary_key=True),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users_table.id', ondelete="CASCADE"), nullable=False),
    Column('refresh_token', VARCHAR(255), unique=True, nullable=False),
    Column('user_agent', VARCHAR(255), nullable=True),
    Column('ip_address', VARCHAR(45), nullable=True),
    Column('created_at', TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP')),
    Column('last_activity', TIMESTAMP(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP')),
    Column('expires_at', TIMESTAMP(timezone=True), nullable=False),
    Column('is_active', Boolean, default=True)
)