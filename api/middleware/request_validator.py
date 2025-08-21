from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging
import json

logger = logging.getLogger("api.middleware.request_validator")

class RequestValidator:
    async def __call__(self, request: Request, call_next):
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if len(body) > 1_000_000:  # 1MB limit
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "El tamaño de la solicitud excede el límite permitido"}
                    )
                header_content_type = request.headers.get("content-type", "").lower()

                if "application/json" in header_content_type:
                    try:
                        if body:
                            _ = json.loads(body)
                    except json.JSONDecodeError:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": "Cuerpo JSON inválido"}
                        )
                elif not ("application/x-www-form-urlencoded" in header_content_type or header_content_type == ""):
                    return JSONResponse(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        content={"detail": "Tipo de contenido no soportado. Use application/json o application/x-www-form-urlencoded"}
                    )
            
            # Validar headers requeridos
            if request.method != "GET":
                required_headers = ["accept", "content-type"]
                for header in required_headers:
                    if header not in request.headers:
                        return JSONResponse(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": f"Falta el header requerido: {header}"}
                        )
            return await call_next(request)
            
        except Exception as e:
            logger.error(f"Error en validación de solicitud: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Error interno del servidor"}
            )
