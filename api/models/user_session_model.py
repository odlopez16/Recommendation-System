from datetime import datetime, timedelta
from pydantic import BaseModel, UUID4, Field
from typing import Optional


class UserSessionBase(BaseModel):
    user_id: UUID4
    refresh_token: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    expires_at: datetime = Field(default_factory=lambda: datetime.now() + timedelta(days=30))
    is_active: bool = True

    class Config:
        from_attributes = True

class UserSessionCreate(UserSessionBase):
    pass


class UserSession(UserSessionBase):
    id: UUID4
    created_at: datetime
    last_activity: datetime