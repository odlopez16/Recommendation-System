from sqlalchemy import Table, Column, Boolean, ForeignKey, DateTime, text, UniqueConstraint, Index
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
        onupdate=text('CURRENT_TIMESTAMP')),
    # Unique constraint to ensure one interaction per user per product
    UniqueConstraint('user_id', 'product_id', name='uq_user_product_interaction'),
    # Performance indexes
    Index('idx_user_interactions_user_liked', 'user_id', 'liked'),
    Index('idx_user_interactions_product_liked', 'product_id', 'liked'),
    Index('idx_user_interactions_created_at', 'created_at')
)
