from pydantic import BaseModel

from api.models.products_model import Product


class Description(BaseModel):
    description: str

class Response(BaseModel):
    answer: str
    products: list[Product]

class NumLikes(BaseModel):
    num_likes: int