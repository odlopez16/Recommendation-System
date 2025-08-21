from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr, Field, UUID4
from typing import Optional


class UserWithoutPassword(BaseModel):
    """
    Modelo para representar un usuario sin información sensible.
    
    Este modelo se utiliza para devolver información del usuario en respuestas API,
    excluyendo datos sensibles como la contraseña.
    
    Attributes:
        id (UUID, opcional): Identificador único del usuario
        email (EmailStr): Dirección de correo electrónico del usuario
        is_active (bool): Estado de activación de la cuenta
        created_at (datetime): Fecha y hora de creación de la cuenta
    
    Example:
        >>> user = UserWithoutPassword(
        ...     email="usuario@ejemplo.com",
        ...     is_active=True
        ... )
    """
    id: UUID4
    email: EmailStr
    is_active: bool = True
    created_at: datetime

class UserInDB(BaseModel):
    """
    Modelo para representar un usuario en la base de datos.
    
    Este modelo contiene todos los campos necesarios para almacenar
    la información del usuario en la base de datos, incluyendo la
    contraseña hasheada.
    
    Attributes:
        id (UUID): Identificador único del usuario
        email (EmailStr): Dirección de correo electrónico del usuario
        hashed_password (str): Contraseña hasheada para almacenamiento seguro
        is_active (bool): Estado de activación de la cuenta
        created_at (datetime): Fecha y hora de creación de la cuenta
    """
    id: UUID4
    email: EmailStr
    hashed_password: str
    is_active: bool
    created_at: datetime
    class Config:
            from_attributes = True

class UserCreate(BaseModel):
    """
    Modelo para la creación de nuevos usuarios.
    
    Este modelo define los campos requeridos para crear un nuevo usuario
    en el sistema. Solo incluye los campos necesarios para el registro.
    
    Attributes:
        email (EmailStr): Dirección de correo electrónico del nuevo usuario
        password (str): Contraseña en texto plano (será hasheada automáticamente)
        is_active (bool, optional): Estado de activación de la cuenta. Por defecto es True.
    """
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    """
    Modelo para la autenticación de usuarios.
    
    Define los campos necesarios para el proceso de inicio de sesión.
    
    Attributes:
        email (EmailStr): Dirección de correo electrónico del usuario
        password (str): Contraseña del usuario
    
    Note:
        La contraseña se verifica contra el hash almacenado en la base de datos.
    """
    email: EmailStr
    password: str
    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    """
    Modelo para la respuesta de inicio de sesión exitoso.
    
    Contiene los tokens necesarios para la autenticación continua
    del usuario en el sistema.
    
    Attributes:
        access_token (str): Token JWT para acceder a recursos protegidos
        token_type (str): Tipo de token, por defecto "Bearer"
        refresh_token (str): Token para renovar el access_token cuando expire
    """
    access_token: str
    token_type: str = "Bearer"
    refresh_token: str

    class Config:
        from_attributes = True


class TokenData(BaseModel):
    """
    Modelo para los datos contenidos en el token JWT.
    
    Contiene la información esencial que se codifica dentro del
    token JWT para identificar al usuario y la sesión.
    
    Attributes:
        email (str): Email del usuario al que pertenece el token
        exp (datetime, opcional): Fecha y hora de expiración del token
        session_id (UUID, opcional): Identificador único de la sesión
    """
    email: str
    exp: Optional[datetime]
    session_id: Optional[UUID4] = None
    class Config:
        from_attributes = True


class UserSession(BaseModel):
    """
    Modelo para gestionar las sesiones de usuario.
    
    Mantiene un registro de las sesiones activas de los usuarios,
    incluyendo información sobre el dispositivo y la actividad.
    
    Attributes:
        id (UUID): Identificador único de la sesión
        user_id (UUID): ID del usuario al que pertenece la sesión
        refresh_token (str): Token de actualización asociado a la sesión
        user_agent (str, opcional): Información del navegador/dispositivo
        ip_address (str, opcional): Dirección IP del cliente
        created_at (datetime): Fecha y hora de creación de la sesión
        last_activity (datetime): Última actividad registrada
        expires_at (datetime): Fecha y hora de expiración de la sesión
        is_active (bool): Estado activo/inactivo de la sesión
    
    Note:
        Una sesión inactiva no permite la generación de nuevos tokens
        de acceso mediante el refresh token asociado.
    """
    id: UUID4
    user_id: UUID4
    refresh_token: str
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    is_active: bool = True
    
    class Config:
        from_attributes = True
