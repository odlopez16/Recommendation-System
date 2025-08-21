from datetime import datetime, timezone
from pydantic import BaseModel, UUID4, Field


class EmbeddingIn(BaseModel):
    """
    Model for data input when creating an embedding.
    The embedding is stored as bytes in the database.
    """
    product_id: UUID4
    embedding: bytes



class Embedding(BaseModel):
    """
    Model for data output, both for creation and extraction.
    The embedding is returned as a list of floats.
    """
    id: UUID4
    embedding: list[float]
    created_at: datetime
    product_id: UUID4

    class Config:
        from_attributes = True