from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4

class RevokedTokenBase(BaseModel):
    token: str
    revoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    user_id: Optional[UUID]

class RevokedTokenCreate(RevokedTokenBase):
    pass

class RevokedToken(RevokedTokenBase):
    id: UUID = Field(default_factory=uuid4)

    class Config:
        from_attributes = True
