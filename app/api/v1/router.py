"""
APIRouter aggregator for API v1.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import router as v1_endpoints
from app.api.v1.auth import router as auth_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(v1_endpoints, tags=["v1"])
