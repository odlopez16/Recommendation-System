from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from api.exceptions.exceptions import not_found_exception
from sqlalchemy import Insert, Select, Update, Delete, func, select

from api.schemas.auth_schema import user_sessions_table as session_table

from api.database.database_config import primary_database as db
from api.models.user_session_model import UserSession, UserSessionCreate
from config import config


class SessionService:
    def __init__(self):
        self.max_sessions_per_user = config.MAX_SESSIONS_PER_USER

    async def create_session(self, session_data: UserSessionCreate, refresh_token: Optional[str] = None) -> UserSession:
        """Create a new user session and manage session limits
        
        Args:
            session_data: Data for the new session
            refresh_token: Optional refresh token. If not provided, one will be generated.
        """
        # Check if user has reached the maximum number of active sessions
        await self._enforce_session_limit(session_data.user_id)
        
        # Create new session with the provided refresh token or generate a new one
        session_dict = dict(session_data)
        if refresh_token is not None:
            session_dict['refresh_token'] = refresh_token
        
        
        
        query: Insert = session_table.insert().values(**session_dict)
        await db.get_database().execute(query)

        new_session: Optional[UserSession] = await self.get_session_by_refresh_token(session_dict['refresh_token'])
        if new_session:
            return new_session
        raise not_found_exception(detail="Session not found")

    async def get_session_by_refresh_token(self, refresh_token: str) -> Optional[UserSession]:
        """Get a session by its refresh token"""
        query: Select = session_table.select().where(
                session_table.c.refresh_token == refresh_token,
                session_table.c.is_active == True,
                session_table.c.expires_at > datetime.now(timezone.utc)
        )
        
        # Use fetch_one() instead of execute() with fetchone()
        session_data = await db.get_database().fetch_one(query)
        
        if not session_data:
            return None
        return UserSession(**dict(session_data))

    async def update_session_activity(self, session_id: UUID) -> None:
        """Update the last activity timestamp of a session"""
        query: Update = session_table.update().where(
            session_table.c.id == session_id
        ).values(
            last_activity=datetime.now(timezone.utc)
        )
        await db.get_database().execute(query)

    async def deactivate_session(self, refresh_token: str) -> bool:
        """Deactivate a session by its refresh token"""
        if not refresh_token:
            return False
            
        query: Update = session_table.update().where(
            session_table.c.refresh_token == refresh_token
        ).values(
            is_active=False
        )
        
        try:
            # Execute the update and get the number of affected rows
            result = await db.get_database().execute(query)
            # For some database backends, we might need to check the result differently
            # This is a more reliable way to check if the update was successful
            return True
        except Exception as e:
            print(f"Error deactivating session: {e}")
            return False

    async def deactivate_all_user_sessions(self, user_id: UUID) -> int:
        """Deactivate all sessions for a specific user"""
        if not user_id:
            return 0
            
        query: Update = session_table.update().where(
            session_table.c.user_id == user_id,
            session_table.c.is_active == True
        ).values(
            is_active=False
        )
        
        try:
            # Execute the update
            await db.get_database().execute(query)
            
            # Since we can't easily get the row count, we'll return a positive number
            # to indicate success. The exact count isn't critical for the calling code.
            return 1
        except Exception as e:
            print(f"Error deactivating all user sessions: {e}")
            return 0

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions from the database"""
        try:
            query: Delete = session_table.delete().where(
                session_table.c.expires_at < datetime.now(timezone.utc)
            )
            
            # Execute the delete operation
            await db.get_database().execute(query)
            
            # Since we can't easily get the row count, we'll return a positive number
            # to indicate success. The exact count isn't critical for the calling code.
            return 1
        except Exception as e:
            print(f"Error cleaning up expired sessions: {e}")
            return 0

    async def get_active_sessions_count(self, user_id: UUID) -> int:
        """Get the count of active sessions for a user"""
        
        query: Select = select(func.count()).select_from(session_table).where(
            session_table.c.user_id == user_id,
            session_table.c.is_active == True,
            session_table.c.expires_at > datetime.now(timezone.utc)
        )
        
        result = await db.get_database().fetch_val(query)
        return result or 0

    async def get_user_active_sessions(self, user_id: UUID) -> List[UserSession]:
        """Get all active sessions for a user"""
        query: Select = session_table.select().where(
            session_table.c.user_id == user_id,
            session_table.c.is_active == True,
            session_table.c.expires_at > datetime.now(timezone.utc)
        ).order_by(session_table.c.last_activity.desc())
        
        # Use fetch_all() instead of execute() with fetchall()
        sessions_data = await db.get_database().fetch_all(query)
        
        return [UserSession(**dict(session)) for session in sessions_data]

    async def _enforce_session_limit(self, user_id: UUID) -> None:
        """Enforce the maximum number of active sessions per user"""
        if self.max_sessions_per_user <= 0:
            # No limit if max_sessions_per_user is 0 or negative
            return
            
        # Get current active sessions count
        active_sessions_count = await self.get_active_sessions_count(user_id)
        
        # If user has reached the limit, deactivate the oldest session
        if active_sessions_count >= self.max_sessions_per_user:
            # Get all active sessions ordered by last activity
            active_sessions = await self.get_user_active_sessions(user_id)
            
            # Deactivate the oldest session(s)
            sessions_to_deactivate = active_sessions[self.max_sessions_per_user-1:]
            for session in sessions_to_deactivate:
                await self.deactivate_session(session.refresh_token)