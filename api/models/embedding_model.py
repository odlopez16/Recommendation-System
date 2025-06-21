from pydantic import BaseModel, UUID4


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
    product_id: UUID4
    embedding: list[float]

    class Config:
        from_attributes = True