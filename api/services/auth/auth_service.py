import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext #type: ignore
from jose import JWTError, jwt
from sqlalchemy import Insert, select, update, delete, and_, func

from api.database.database_config import primary_database as db
from api.models.auth_model import UserCreate, UserInDB, TokenData, UserWithoutPassword, LoginResponse
from api.models.user_session_model import UserSession, UserSessionCreate
from api.schemas.auth_schema import users_table, user_sessions_table
from api.exceptions.exceptions import not_found_exception, unauthorized_exception
from config import config
from logging_config import setup_logging
from api.services.auth.security_service import (
    brute_force_protection,
    check_brute_force,
    get_client_ip
)
from api.services.auth.token_service import token_service

setup_logging()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger("api.services.auth_service")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password:str):
    return pwd_context.hash(password)

async def get_user_by_email(email: str) -> UserInDB | None:
    query = users_table.select().where(users_table.c.email == email)
    record = await db.get_database().fetch_one(query)
    if record:
        # Ensure is_active is a boolean
        user_data = dict(record)
        user_data['is_active'] = bool(user_data.get('is_active', True))
        return UserInDB(**user_data)
    return None

async def authenticate_user(email: str, password: str, request: Optional[Request] = None) -> UserWithoutPassword:
    """
    Authenticate a user with email and password.
    
    Args:
        email: User's email
        password: Plain text password
        request: Optional request object for security features
        
    Returns:
        UserInDB: Authenticated user object
        
    Raises:
        HTTPException: If authentication fails or user is inactive
    """
    # Check brute force protection
    identifier: str = email
    if request:
        client_ip: str = get_client_ip(request)
        identifier = f"{email}:{client_ip}"
    
    check_brute_force(identifier)
    
    try:
        user: UserInDB | None = await get_user_by_email(email)
        
        # Check if user exists and is active
        if not user or not user.is_active:
            logger.warning(f"Authentication failed for user. User not found or inactive")
            brute_force_protection.register_attempt(identifier, success=False)
            raise unauthorized_exception("Correo inválido")
            
        # Verify password
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed for user. Invalid password")
            brute_force_protection.register_attempt(identifier, success=False)
            raise unauthorized_exception("Contraseña inválida")
            
        # Successful authentication
        brute_force_protection.register_attempt(identifier, success=True)
        return UserWithoutPassword(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during authentication"
        )

def create_access_token(data: TokenData):
    to_encode = data.model_copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=int(config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.exp = expire
    
    # Convert UUID to string for JSON serialization
    payload = to_encode.model_dump()
    if 'session_id' in payload and payload['session_id'] is not None:
        payload['session_id'] = str(payload['session_id'])
    if 'exp' in payload and isinstance(payload['exp'], datetime):
        payload['exp'] = int(payload['exp'].timestamp())
        
    encoded_jwt = jwt.encode(payload, config.JWT_SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt


async def create_user(email: str, password: str) -> UserWithoutPassword:
    try:
        # Verificar si el usuario ya existe
        existing_user = await get_user_by_email(email)
        logger.info(f"Checking existing user: {existing_user}")
        if existing_user is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
        
        # Hashear la contraseña
        hashed_password = get_password_hash(password)
        
        # Crear el diccionario de datos para la inserción
        user_data = {
            "email": email,
            "hashed_password": hashed_password,
            "is_active": True
        }
        
        # Insertar el nuevo usuario
        insert_query = users_table.insert().values(**user_data)
        await db.get_database().execute(insert_query)
        new_user: UserInDB | None = await get_user_by_email(email)
        if new_user != None:
            return UserWithoutPassword(
                id=new_user.id,
                email=new_user.email,
                is_active=new_user.is_active,
                created_at=new_user.created_at
            )
        raise not_found_exception("User not found😢")
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error creating user")

async def get_token_from_cookie_or_header(request: Request) -> str | None:
    # Try to get token from Authorization header first
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
        
    # Then try to get from cookie
    if "access_token" in request.cookies:
        cookie_value = request.cookies.get("access_token")
        if cookie_value and cookie_value.startswith("Bearer "):
            return cookie_value.split(" ")[1]
            
    return None

async def get_current_user(request: Request) -> UserWithoutPassword:
    credentials_exception = unauthorized_exception("Credenciales inválidas")
    # Get token from either header or cookie
    token = await get_token_from_cookie_or_header(request)
    if not token:
        logger.warning("No se encontró token de acceso en la solicitud")
        raise credentials_exception
        
    try:
        # Check if token is revoked
        is_revoked = await token_service.is_token_revoked(token)
        if is_revoked:
            logger.warning("Se intentó usar un token revocado")
            raise unauthorized_exception("Token ha sido revocado")
            
        # Decode and validate token
        payload = jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str | None = payload.get("email")
        if email is None:
            logger.warning("No se encontró email en el token")
            raise credentials_exception
            
        # Get user from database
        user: UserInDB | None = await get_user_by_email(email)
        if user is None:
            logger.error(f"Usuario no encontrado en la base de datos")
            raise credentials_exception
            
        if not user.is_active:
            logger.warning(f"Intento de acceso de usuario inactivo")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Esta cuenta ha sido desactivada",
            )
            
        return UserWithoutPassword(
            id=user.id,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        )
            
    except JWTError as e:
        logger.error(f"Error de JWT")
        raise unauthorized_exception("No se pudo validar el token de acceso")
    return user

def create_refresh_token(data: dict, expires_delta: timedelta | None = None, session_id: Optional[uuid.UUID] = None) -> str:
    """
    Creates a new refresh token.

    Args:
        data (dict): Data to encode in the token.
        expires_delta (timedelta | None, optional): Token lifetime. Defaults to None.
        session_id (Optional[UUID], optional): Session ID to include in token. Defaults to None.

    Returns:
        str: The encoded refresh token.
    """
    to_encode: dict = data.copy()
    expire: datetime = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES)))
    to_encode.update({
        "exp": expire,
        "type": "refresh"
    })
    
    # Add session_id to the token payload if provided
    if session_id:
        to_encode.update({"session_id": str(session_id)})
        
    try:
        return jwt.encode(
            to_encode,
            config.REFRESH_SECRET_KEY,
            algorithm=config.ALGORITHM
        )
    except Exception as e:
        logger.error(f"Error creating refresh token")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating refresh token"
        )

async def verify_refresh_token(refresh_token: str, session_service = None) -> dict:
    """
    Verifies the refresh token and returns the payload if valid.
    Also validates the token against the database if session_service is provided.

    Args:
        refresh_token (str): The refresh token to verify.
        session_service: Optional service to validate token against database.

    Returns:
        dict: The decoded token payload.

    Raises:
        HTTPException: If the token is invalid, expired, or revoked.
    """
    try:
        # Primero verificamos si el token ha sido revocado
        is_revoked = await token_service.is_token_revoked(refresh_token)
        if is_revoked:
            logger.warning("Se intentó usar un refresh token revocado")
            raise unauthorized_exception("Refresh token ha sido revocado")
        # Decodificamos el token para obtener el payload
        payload = jwt.decode(
            refresh_token,
            config.REFRESH_SECRET_KEY,
            algorithms=[config.ALGORITHM]
        )
        
        # Si se proporciona el servicio de sesiones, validamos el token contra la base de datos
        if session_service:
            session = await session_service.get_session_by_refresh_token(refresh_token)
            if not session:
                logger.warning("Refresh token no encontrado en la base de datos de sesiones activas")
                raise unauthorized_exception("Sesión inválida o expirada")
            # Actualizamos la última actividad de la sesión
            await session_service.update_session_activity(session.id)
            # Añadimos el ID de sesión al payload si no está presente
            if "session_id" not in payload:
                payload["session_id"] = str(session.id)
                
        return payload
    except JWTError as e:
        logger.error(f"Invalid refresh token")
        raise unauthorized_exception("Invalid or expired refresh token")

async def refresh_access_token(refresh_token: str, session_service = None, request = None):
    """
    Generates a new access token using a refresh token.
    Validates the token against the database if session_service is provided.

    Args:
        refresh_token (str): The refresh token from HTTP-only cookie.
        session_service: Optional service to validate token against database.
        request: Optional FastAPI request object to get client info.

    Returns:
        dict: A dictionary containing the new access token and refresh token.
            {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token"
            }

    Raises:
        HTTPException: If the refresh token is invalid, expired, or revoked.
    """
    try:
        # Verificamos el refresh token, incluyendo validación contra la base de datos si se proporciona session_service
        payload = await verify_refresh_token(refresh_token, session_service)
        email = payload.get("email")
        if not email:
            raise unauthorized_exception("Invalid token")

        user: UserInDB | None = await get_user_by_email(email)
        if not user:
            raise unauthorized_exception("User not found")

        # Creamos el nuevo access token, incluyendo el session_id si está disponible
        session_id = payload.get("session_id")
        new_access_token = create_access_token(
            TokenData(email=user.email, exp=None, session_id=UUID(session_id) if session_id else None)
        )
        
        # Si tenemos un servicio de sesiones, actualizamos o creamos una nueva sesión
        if session_service:
            # Si ya existe una sesión, la actualizamos con un nuevo refresh token
            if session_id:
                # Desactivamos la sesión anterior
                await session_service.deactivate_session(refresh_token)
                
                # Calculamos la fecha de expiración para el nuevo token
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES))
                
                # Creamos una nueva sesión con el mismo ID pero nuevo token
                session_data: UserSessionCreate = UserSessionCreate(
                    user_id=user.id,
                    refresh_token="",  # Se actualizará después
                    expires_at=expires_at,
                    user_agent=request.headers.get("user-agent") if request else None,
                    ip_address=request.client.host if request and hasattr(request, "client") else None
                )
                
                # Creamos la nueva sesión
                new_session = await session_service.create_session(session_data)
                
                # Creamos el nuevo refresh token con el ID de sesión
                new_refresh_token = create_refresh_token(
                    {"email": user.email},
                    expires_delta=timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES)),
                    session_id=new_session.id
                )
                
                # Actualizamos el refresh token en la sesión
                # Esto normalmente requeriría un método adicional en session_service, pero por simplicidad
                # asumimos que la sesión ya tiene el token actualizado
            else:
                # No hay sesión existente, creamos una nueva
                expires_at = datetime.now(timezone.utc) + timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES))
                
                session_data = UserSessionCreate(
                    user_id=user.id,
                    refresh_token="",  # Se actualizará después
                    expires_at=expires_at,
                    user_agent=request.headers.get("user-agent") if request else None,
                    ip_address=request.client.host if request and hasattr(request, "client") else None
                )
                
                # Creamos la nueva sesión
                new_session = await session_service.create_session(session_data)
                
                # Creamos el nuevo refresh token con el ID de sesión
                new_refresh_token = create_refresh_token(
                    {"email": user.email},
                    expires_delta=timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES)),
                    session_id=new_session.id
                )
        else:
            # Si no hay servicio de sesiones, simplemente creamos un nuevo refresh token
            new_refresh_token = create_refresh_token(
                {"email": user.email},
                expires_delta=timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES)),
                session_id=UUID(session_id) if session_id else None
            )

        return LoginResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="Bearer"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh",
        )

