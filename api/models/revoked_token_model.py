from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field, UUID4
from typing import Optional
    

class RevokedTokenCreate(BaseModel):
    token: str
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))
    user_id: Optional[UUID4]

class RevokedToken(BaseModel):
    id: UUID4
    token: str
    revoked_at: datetime
    expires_at: datetime
    user_id: Optional[UUID4]

    class Config:
        from_attributes = True
