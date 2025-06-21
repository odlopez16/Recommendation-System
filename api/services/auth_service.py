import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from sqlalchemy import Insert
from api.models.auth_model import UserInDB, TokenData
from api.database.database_config import primary_database as db
import uuid
from fastapi import HTTPException, status, Depends
from api.schemas.auth_schema import users_table
from config import config
import logging
from logging_config import setup_logging
from fastapi.security import OAuth2PasswordBearer

setup_logging()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logger = logging.getLogger("api.services.auth_service")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password:str):
    return pwd_context.hash(password)

async def get_user_by_email(email: str):
    query = users_table.select().where(users_table.c.email == email)
    record = await db.get_database().fetch_one(query)
    if record:
        return UserInDB(**dict(record))
    logger.error(f"User with email {email} doesn't exist")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail= f"User with email {email} does'nt exist")

async def authenticate_user(email: str, password: str):
    user = await get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: TokenData):
    to_encode = data.model_copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=int(config.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.exp = expire
    encoded_jwt = jwt.encode(dict(to_encode), config.JWT_SECRET_KEY, algorithm=config.ALGORITHM)
    return encoded_jwt


async def create_user(email: str, password: str):
    hashed_password = get_password_hash(password)
    insert_query: Insert = users_table.insert().values(
        **dict(UserInDB(id=uuid.uuid4(),
                        email=email,
                        hashed_password=hashed_password,
                        is_active=True)
                )
    )
    await db.get_database().execute(insert_query)
    logger.info(f"User created with email {email}")
    return await get_user_by_email(email)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload: dict= jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.ALGORITHM])
        email: str = payload["email"]
        expire: datetime = payload["exp"]
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email, exp=expire)
    except JWTError:
        logger.error("JWT error")
        raise credentials_exception
    user: UserInDB = await get_user_by_email(token_data.email)
    if user is None:
        logger.error("User not found")
        raise credentials_exception
    return user
