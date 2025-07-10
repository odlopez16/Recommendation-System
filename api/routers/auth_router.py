import sys
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any
from datetime import timedelta, datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Cookie
from fastapi.security import OAuth2PasswordRequestForm
from jose import jwt  # type: ignore
from config import config
from logging_config import setup_logging
from api.models.auth_model import TokenData, UserCreate, LoginResponse, UserInDB, UserWithoutPassword
from api.models.user_session_model import UserSession, UserSessionCreate
from api.exceptions.exceptions import unauthorized_exception
from api.services.token_service import token_service
from api.services.session_service import SessionService
from api.services.auth_service import (
    authenticate_user, 
    create_access_token, 
    create_refresh_token,
    refresh_access_token as refresh_tokens,
    create_user, 
    get_current_user,
    get_user_by_email
)


logger = logging.getLogger("api.auth_router")
setup_logging()

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserWithoutPassword, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    db_user: UserWithoutPassword | None = await create_user(user.email, user.password)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User not created")
    return db_user

@router.post("/login", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def login(
    request: Request,
    response: Response, 
    form_data: OAuth2PasswordRequestForm = Depends(),
    session_service: SessionService = Depends()
):
    try:
        db_user: UserInDB = await authenticate_user(
            email=form_data.username,
            password=form_data.password,
            request=request
        )

        # Verificar si el usuario está activo
        if not db_user.is_active:
            logger.warning(f"Login attempt for inactive user: {db_user.email}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This account has been deactivated"
            )

        # Crear una nueva sesión para el usuario
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES))
        
        # Crear refresh token
        refresh_token = create_refresh_token(
            {"email": db_user.email},
            expires_delta=timedelta(minutes=int(config.REFRESH_TOKEN_EXPIRE_MINUTES))
        )
        
        # Crear la sesión con el refresh token
        session_data = UserSessionCreate(
            user_id=db_user.id,  # type: ignore
            refresh_token=refresh_token,
            expires_at=expires_at,
            user_agent=request.headers.get("user-agent"),
            ip_address=request.client.host if hasattr(request, "client") and request.client is not None else None
        )
        
        # Crear la sesión en la base de datos
        session = await session_service.create_session(session_data, refresh_token=refresh_token)
        
        # Crear access token con el ID de sesión
        access_token = create_access_token(
            TokenData(email=db_user.email, exp=None, session_id=session.id)
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions with proper logging
        logger.warning(f"Login failed for user {form_data.username}: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login for {form_data.username}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        )

    # Configurar las cookies
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,
        secure=config.ENV_STATE == "production",
        samesite="strict",
        max_age=60 * int(config.ACCESS_TOKEN_EXPIRE_MINUTES),
        path="/"
    )
    
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=config.ENV_STATE == "production",
        samesite="strict",
        max_age=60 * int(config.REFRESH_TOKEN_EXPIRE_MINUTES),
        path="/"
    )

    # Devolver los tokens en la respuesta
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer"
    )

@router.post("/refresh_token", response_model=LoginResponse, status_code=status.HTTP_200_OK)
async def refresh_token(
    request: Request, 
    response: Response,
    refresh_token: str = Cookie(..., alias="refresh_token"),
    session_service: SessionService = Depends()
):
    """
    Refresh an access token using a valid refresh token from HTTP-only cookie.
    Returns a new access token and sets a new refresh token in HTTP-only cookie.
    
    This endpoint will revoke the old refresh token after issuing a new one.
    Validates the refresh token against the database using the session service.
    """
    try:
        # Verificar si el token de refresco ha sido revocado
        if await token_service.is_token_revoked(refresh_token):
            logger.warning("Intento de refresco con token revocado")
            response.delete_cookie("refresh_token", path="/")
            raise unauthorized_exception("Refresh token revocado")
        
        try:
            # Verificar la sesión asociada al token
            session = await session_service.get_session_by_refresh_token(refresh_token)
            if not session or not session.is_active:
                logger.warning("Intento de refresco con sesión inválida o inactiva")
                response.delete_cookie("refresh_token", path="/")
                raise unauthorized_exception("Sesión inválida o expirada")
                
            # Verificar el payload del token
            payload: dict[str, Any] = jwt.get_unverified_claims(refresh_token)
            email = payload.get("email")
            if not email:
                raise unauthorized_exception("Token inválido")
                
            user = await get_user_by_email(email)
            if not user:
                raise unauthorized_exception("Usuario no encontrado")
                
            # Verificar que el usuario del token coincida con el usuario de la sesión
            if str(user.id) != str(session.user_id):
                logger.warning(f"Mismatch entre usuario del token ({user.id}) y usuario de la sesión ({session.user_id})")
                response.delete_cookie("refresh_token", path="/")
                raise unauthorized_exception("Token inválido para esta sesión")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error al verificar el token de refresco: {str(e)}")
            raise unauthorized_exception("Token de refresco inválido")
        
        # Generar nuevos tokens, pasando el servicio de sesiones y la solicitud
        tokens: LoginResponse = await refresh_tokens(refresh_token, session_service, request)
        
        # Revocar el token de refresco anterior
        await token_service.revoke_token(refresh_token, user)
        
        # Desactivar la sesión anterior
        await session_service.deactivate_session(refresh_token)
        
        # Configurar la nueva cookie de refresh token
        response.set_cookie(
            key="refresh_token",
            value=tokens.refresh_token,
            httponly=True,
            secure=config.ENV_STATE == "prod",
            samesite="strict",
            max_age=60 * int(config.REFRESH_TOKEN_EXPIRE_MINUTES),
            path="/"
        )
        
        return LoginResponse(
            access_token=tokens.access_token,
            refresh_token="",  # No devolver el refresh token en el cuerpo
            token_type="Bearer"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error al refrescar el token: {str(e)}")
        response.delete_cookie("refresh_token", path="/")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al refrescar el token"
        )

@router.get("/me", response_model=UserInDB, status_code=status.HTTP_200_OK)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user


@router.get("/sessions", status_code=status.HTTP_200_OK)
async def get_active_sessions(
    current_user: UserInDB = Depends(get_current_user),
    session_service: SessionService = Depends()
):
    """
    Obtiene todas las sesiones activas del usuario actual.
    """
    try:
        sessions = await session_service.get_user_active_sessions(current_user.id)# type: ignore
        
        # Transformar las sesiones a un formato más amigable para la respuesta
        sessions_data = [
            {
                "id": str(session.id),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "user_agent": session.user_agent,
                "ip_address": session.ip_address,
                "expires_at": session.expires_at.isoformat()
            }
            for session in sessions
        ]
        
        return {
            "count": len(sessions_data),
            "sessions": sessions_data
        }
    except Exception as e:
        logger.error(f"Error al obtener sesiones para {current_user.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al obtener las sesiones activas"
        )

@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    current_user: UserInDB = Depends(get_current_user),
    session_service: SessionService = Depends()
):
    """
    Cierra la sesión del usuario revocando el token de refresco y desactivando la sesión.
    """
    # Obtener el token de refresco de la cookie
    refresh_token = request.cookies.get("refresh_token")
    
    if refresh_token:
        try:
            # Revocar el token de refresco
            await token_service.revoke_token(refresh_token, current_user)
            
            # Desactivar la sesión asociada al token
            await session_service.deactivate_session(refresh_token)
            
            # Opcionalmente, podríamos desactivar todas las sesiones del usuario
            # si se quiere cerrar sesión en todos los dispositivos
            # await session_service.deactivate_all_user_sessions(current_user.id)
            
            logger.info(f"Usuario {current_user.email} cerró sesión exitosamente")
        except Exception as e:
            logger.error(f"Error al cerrar sesión para {current_user.email}: {str(e)}")
    
    for cookie_name in ["access_token", "refresh_token"]:
        response.delete_cookie(
            key=cookie_name,
            path="/",
            domain=None,
            secure=config.ENV_STATE == "production",
            httponly=True,
            samesite="strict"
        )
    
    return None
    