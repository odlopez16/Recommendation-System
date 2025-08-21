from datetime import datetime
from pydantic import BaseModel,UUID4
    

class UserProductInteractionCreate(BaseModel):
    product_id: UUID4
    liked: bool = True

class UserProductInteractionUpdate(BaseModel):
    liked: bool

class UserProductInteractionInDB(BaseModel):
    id: UUID4
    user_id: UUID4
    product_id: UUID4
    liked: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
