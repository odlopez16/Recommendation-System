from pydantic import BaseModel, UUID4


class Product(BaseModel):
    """
    Model for product output data, including the product ID.
    """
    id: UUID4
    name: str
    price: float
    description: str

    class Config:
        from_attributes = True