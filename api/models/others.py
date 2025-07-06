from pydantic import BaseModel


class Description(BaseModel):
    description: str

class Response(BaseModel):
    answer: str
    products: list