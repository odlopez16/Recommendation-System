import sys
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal
from fastapi import APIRouter, HTTPException, status, Depends
from api.services.auth_service import authenticate_user, create_access_token, create_user, get_current_user
from api.models.auth_model import TokenData, UserCreate, UserLogin, Token, UserInDB
import logging
from logging_config import setup_logging


setup_logging()

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    db_user = await create_user(user.email, user.password)
    return db_user

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(user: UserLogin):
    db_user: UserInDB | Literal[False] = await authenticate_user(user.email, user.password)
    if not db_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    access_token = create_access_token(
        data=TokenData(email=db_user.email, exp=None)
    )
    return Token(access_token=access_token, token_type="bearer")

@router.get("/me", response_model=UserInDB, status_code=status.HTTP_200_OK)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user

