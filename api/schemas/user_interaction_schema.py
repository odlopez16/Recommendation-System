from sqlalchemy import Table, Column, Boolean, ForeignKey, DateTime, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from api.database.database_config import metadata

# SQLAlchemy table for user interactions
user_interactions_table = Table(
    'user_interactions_table',
    metadata,
    Column('id', PG_UUID(as_uuid=True), primary_key=True, server_default=text('gen_random_uuid()')),
    Column('user_id', PG_UUID(as_uuid=True), ForeignKey('users_table.id', ondelete='CASCADE'), nullable=False, index=True),
    Column('product_id', PG_UUID(as_uuid=True), ForeignKey('products_table.id', ondelete='CASCADE'), nullable=False, index=True),
    Column('liked', Boolean, nullable=False, default=True),
    Column('created_at', DateTime(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP')),
    Column('updated_at', DateTime(timezone=True), nullable=False, server_default=text('CURRENT_TIMESTAMP'), 
        onupdate=text('CURRENT_TIMESTAMP'))
    # Add a unique constraint to ensure one like/dislike per user per product
    # This will be added in a separate migration
)
