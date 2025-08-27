import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest
import psutil

# Métricas de Prometheus
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
MEMORY_USAGE = Histogram('memory_usage_percent', 'Memory usage percentage')

logger = logging.getLogger("api.middleware.performance")

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware para monitoreo de performance y métricas"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Monitorear memoria antes de la request
        memory_before = psutil.virtual_memory().percent
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"Request failed: {e}")
            status_code = 500
            response = Response("Internal Server Error", status_code=500)
        
        # Calcular duración
        duration = time.time() - start_time
        
        # Monitorear memoria después
        memory_after = psutil.virtual_memory().percent
        
        # Registrar métricas
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()
        
        REQUEST_DURATION.observe(duration)
        MEMORY_USAGE.observe(memory_after)
        
        # Log requests lentas
        if duration > 2.0:  # > 2 segundos
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.2f}s, memory: {memory_before}% -> {memory_after}%"
            )
        
        # Agregar headers de performance
        response.headers["X-Process-Time"] = str(duration)
        response.headers["X-Memory-Usage"] = f"{memory_after}%"
        
        return response

async def metrics_endpoint():
    """Endpoint para métricas de Prometheus"""
    return Response(generate_latest(), media_type="text/plain")