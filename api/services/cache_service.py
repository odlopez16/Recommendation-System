import asyncio
from functools import lru_cache
from typing import Optional
import redis.asyncio as redis
from databases import Database
import logging

logger = logging.getLogger("api.cache")

class CacheManager:
    """Gestor de caché centralizado con Redis y memoria local"""
    
    def __init__(self):
        self._redis: Optional[redis.Redis] = None  # type: ignore
        self._local_cache = {}
        
    async def get_redis(self) -> redis.Redis:
        if not self._redis:
            self._redis = redis.from_url("redis://localhost:6379", decode_responses=True)
        if self._redis is None:
            raise RuntimeError("Redis connection not initialized")
        return self._redis
    
    @lru_cache(maxsize=1000)
    async def get_cached_embeddings(self) -> list:
        """Cache embeddings en memoria para acceso rápido"""
        cache_key = "embeddings:all"
        redis_client = await self.get_redis()
        
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return eval(cached)  # En producción usar pickle/json
        except Exception:
            pass
            
        return []
    
    async def set_cached_embeddings(self, embeddings: list, ttl: int = 3600):
        """Guarda embeddings en caché con TTL"""
        cache_key = "embeddings:all"
        redis_client = await self.get_redis()
        
        try:
            await redis_client.setex(cache_key, ttl, str(embeddings))
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

cache_manager = CacheManager()