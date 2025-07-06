import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from api.database.database_config import initialize_databases, close_databases
from api.routers.auth_router import router as auth_router
from api.routers.embedding_router import router as embed_router
from api.routers.products_router import router as products_router
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
        await initialize_databases()
        logger.info("Databases initialized successfully.")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down application and closing databases...")
        await close_databases()
        logger.info("Databases closed successfully.")

app = FastAPI(lifespan=lifespan)

# Configuración CORS segura
origins = [
    "http://localhost:3000",
    "http://localhost:4000",
    "http://localhost:3001",
    # Agrega aquí los dominios de producción cuando los tengas
]

# Middleware para manejar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Set-Cookie", "Authorization"],
    max_age=3600,
)



app.include_router(embed_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(products_router, prefix="/api")

@app.get("/home")
async def home():
    return {"message": "Welcome to the API"}


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
