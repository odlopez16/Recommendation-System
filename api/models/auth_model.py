from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional
from uuid import UUID

class UserInDB(BaseModel):
    id: Optional[UUID]
    email: EmailStr
    hashed_password: str
    is_active: bool = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: str
    exp: Optional[datetime]
