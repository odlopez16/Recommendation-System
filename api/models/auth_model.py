from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID, uuid4


class UserWithoutPassword(BaseModel):
    id: Optional[UUID] = None
    email: EmailStr
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserInDB(BaseModel):
    id: UUID
    email: EmailStr
    hashed_password: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    class Config:
            from_attributes = True

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    refresh_token: str

    class Config:
        from_attributes = True


class TokenData(BaseModel):
    email: str
    exp: Optional[datetime]
    session_id: Optional[UUID] = None
    class Config:
        from_attributes = True


class UserSession(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    refresh_token: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True
