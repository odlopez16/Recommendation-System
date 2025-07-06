from typing import Optional
from pydantic import BaseModel, UUID4
from datetime import datetime


class Product(BaseModel):
    """
    Model for product output data, including the product ID.
    """
    id: UUID4
    name: str
    price: float
    description: str
    image: Optional[str]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True