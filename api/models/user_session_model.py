from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4


class UserSessionBase(BaseModel):
    user_id: UUID
    refresh_token: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    expires_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

class UserSessionCreate(UserSessionBase):
    pass


class UserSession(UserSessionBase):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))