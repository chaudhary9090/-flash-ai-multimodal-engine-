"""
Authentication Routes: Register & Login (JWT Token)
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.models.db import db
from app.core.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str


@router.post("/register", status_code=status.HTTP_210_CREATED if hasattr(status, "HTTP_210_CREATED") else status.HTTP_201_CREATED)
def register(request: AuthRequest):
    success = db.register_user(request.username, request.password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists.")
    return {"message": f"User '{request.username}' successfully registered."}


@router.post("/login", response_model=TokenResponse)
def login(request: AuthRequest):
    authenticated = db.authenticate_user(request.username, request.password)
    if not authenticated:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    token = create_access_token(data={"sub": request.username})
    return TokenResponse(access_token=token, username=request.username)
