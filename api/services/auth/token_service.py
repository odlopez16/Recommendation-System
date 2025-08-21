import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from databases import Database
from jose import jwt
from sqlalchemy import Insert, Select, Delete
from fastapi import HTTPException, status

from api.database.database_config import primary_database as db
from api.models.revoked_token_model import RevokedTokenCreate
from api.schemas.auth_schema import revoked_tokens_table
from api.models.auth_model import UserInDB, UserWithoutPassword

logger = logging.getLogger("api.services.token_service")

class TokenService:
    def __init__(self):
        self.db: Database = db.get_database()

    async def revoke_token(
        self, 
        token: str, 
        user: UserInDB | UserWithoutPassword,
        expires_at: Optional[datetime] = None
    ) -> None:
        """
        Agrega un token a la lista de revocados.
        
        Args:
            token: El token a revocar
            user: El usuario dueño del token
            expires_at: Fecha de expiración del token (si no se proporciona, se calcula)
        """
        if not expires_at:
            # Obtener la expiración del token si no se proporciona
            try:
                payload = jwt.get_unverified_claims(token)
                expires_at = datetime.fromtimestamp(payload["exp"])
            except Exception as e:
                logger.warning(f"No se pudo obtener la expiración del token")
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)

        revoked_token = RevokedTokenCreate(
            token=token,
            user_id=user.id
        )

        try:
            query: Insert = revoked_tokens_table.insert().values(
                token=revoked_token.token,
                expires_at=revoked_token.expires_at,
                user_id=revoked_token.user_id
            )
            await self.db.execute(query)
            
            logger.info(f"Token revocado para el usuario {user.email}")
        except Exception as e:
            logger.error(f"Error al revocar el token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error al procesar la revocación del token"
            )

    async def is_token_revoked(self, token: str) -> bool:
        """
        Verifica si un token ha sido revocado.
        
        Args:
            token: El token a verificar
            
        Returns:
            bool: True si el token está revocado, False en caso contrario
        """
        try:
            query: Select = revoked_tokens_table.select().where(
                revoked_tokens_table.c.token == token
            )
            result = await self.db.fetch_one(query)
            return result is not None
        except Exception as e:
            logger.error(f"Error al verificar token revocado")
            return False

    async def cleanup_expired_tokens(self) -> None:
        """
        Elimina los tokens expirados de la base de datos.
        """
        try:
            query: Delete = revoked_tokens_table.delete().where(
                revoked_tokens_table.c.expires_at < datetime.now(timezone.utc)
            )
            await self.db.execute(query)
            logger.info("Tokens expirados limpiados")
        except Exception as e:
            logger.error(f"Error al limpiar tokens expirados")

# Instancia global para usar en toda la aplicación
token_service = TokenService()
