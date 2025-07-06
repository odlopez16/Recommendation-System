from fastapi import HTTPException, status


unauthorized_exception = lambda detail: HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail=detail,
    headers={"WWW-Authenticate": "Bearer"}
)
    