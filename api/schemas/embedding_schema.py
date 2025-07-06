import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import Table, Column, text
from api.database.database_config import metadata
from sqlalchemy.dialects.postgresql import DATE, UUID, BYTEA


embeddings_table = Table(
    'embeddings_table',
    metadata,
    Column('id', UUID, primary_key=True, server_default=text('gen_random_uuid()')),
    Column('product_id', UUID,unique=True, nullable=False),
    Column('embedding', BYTEA, nullable=False),
    Column('created_at', DATE, nullable=False, server_default=text('CURRENT_TIMESTAMP'))
)
