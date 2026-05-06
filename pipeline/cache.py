import json
import logging
import os
import redis.asyncio as redis
from typing import Any, Optional

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
redis_client: Optional[redis.Redis] = None

async def init_redis():
    """Initialize the Redis connection pool."""
    global redis_client
    if not redis_client:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info(f"Connected to Redis at {REDIS_URL}")

async def close_redis():
    """Close the Redis connection pool."""
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None

async def get_cached(key: str) -> Optional[Any]:
    """Get a JSON-deserialized value from Redis."""
    if not redis_client:
        return None
    try:
        data = await redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.warning(f"Redis get error for {key}: {e}")
        return None

async def set_cached(key: str, value: Any, ttl: int = 3600):
    """Set a JSON-serializable value in Redis with a TTL (default 1 hour)."""
    if not redis_client:
        return
    try:
        await redis_client.set(key, json.dumps(value), ex=ttl)
    except Exception as e:
        logger.warning(f"Redis set error for {key}: {e}")
