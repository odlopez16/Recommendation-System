from os import close
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import uvicorn
from api.database.database_config import initialize_databases, close_databases
from api.routers.auth_router import router as auth_router
from api.routers.embedding_router import router as embed_router
from api.routers.products_router import router as products_router
from api.routers.likes_router import router as likes_router
# Importar middlewares
from api.middleware.rate_limiter import RateLimiter
import asyncio
from api.middleware.request_validator import RequestValidator
import logging
from logging_config import setup_logging


setup_logging()
logger = logging.getLogger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    Initializes and closes database connections on startup and shutdown.
    """
    logger.info("Starting up application and initializing databases...")
    try:
        # Intentar inicializar la base de datos con reintentos
        max_retries = 3
        retry_delay = 2  # segundos
        for attempt in range(max_retries):
            try:
                await initialize_databases()
                logger.info("Databases initialized successfully.")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to initialize databases after {max_retries} attempts")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        logger.info("Shutting down application and closing databases...")
        try:
            await close_databases()
            logger.info("Databases closed successfully.")
        except Exception as e:
            logger.error(f"Error closing databases: {str(e)}")

app = FastAPI(
    title="Recommendation System API",
    description="API para sistema de recomendaciones",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",  # Mover Swagger a /api/docs
    redoc_url="/api/redoc",  # Mover ReDoc a /api/redoc
    openapi_url="/api/openapi.json"  # Mover OpenAPI schema
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

app.middleware("http")(RateLimiter(requests_per_minute=60))
app.middleware("http")(RequestValidator())


origins = [
    "http://localhost:3000",
    "http://localhost:4000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With"
    ],
    expose_headers=["Content-Type", "Authorization"],
    max_age=3600,
)

# Configuración de versiones de API
API_V1_PREFIX = "/api/v1"

# Agregar routers con prefijo de versión
app.include_router(embed_router, prefix=API_V1_PREFIX)
app.include_router(auth_router, prefix=API_V1_PREFIX)
app.include_router(products_router, prefix=API_V1_PREFIX)
app.include_router(likes_router, prefix=API_V1_PREFIX)

# Endpoint para verificar versión de API
@app.get("/api/version")
async def get_api_version():
    return {"version": "1.0.0", "status": "stable"}

@app.get("/home")
async def home():
    return {"message": "Welcome to the API ⚡"}


if __name__ == "__main__":
    port = 4000
    print(f"Starting server on http://localhost:{port}")
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
