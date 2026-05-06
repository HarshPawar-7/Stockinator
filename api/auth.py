from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    # Fetch dynamically to ensure it picks up env vars correctly after startup
    valid_keys = set(k.strip() for k in os.environ.get("API_KEYS", "").split(","))
    if api_key not in valid_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key
