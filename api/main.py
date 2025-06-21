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

app = FastAPI(lifespan=lifespan, prefix= "/api")

# Middleware to handle CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(embed_router)
app.include_router(auth_router)

@app.get("/home")
async def home():
    return {"message": "Welcome to the API"}


if __name__ == "main":
    uvicorn.run(app=app, host="127.0.0.1", port=4000, reload=True)
