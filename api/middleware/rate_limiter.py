from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Tuple
import time
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("api.middleware.rate_limiter")

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}  # IP -> list of timestamps
        
    async def __call__(self, request: Request, call_next):
        # Obtener IP del cliente
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        # Rutas que queremos limitar (auth endpoints)
        rate_limited_paths = ["/api/auth/login", "/api/auth/register"]
        
        if path in rate_limited_paths:
            # Verificar rate limit
            if await self._is_rate_limited(client_ip):
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Demasiadas solicitudes. Por favor, intente de nuevo más tarde."}
                )
        
        # Continuar con la solicitud
        return await call_next(request)
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Inicializar lista de timestamps para esta IP
        if client_ip not in self.requests:
            self.requests[client_ip] = []
        
        # Limpiar timestamps antiguos
        self.requests[client_ip] = [
            ts for ts in self.requests[client_ip] if ts > minute_ago
        ]
        
        # Verificar límite
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return True
        
        # Agregar nuevo timestamp
        self.requests[client_ip].append(current_time)
        return False
