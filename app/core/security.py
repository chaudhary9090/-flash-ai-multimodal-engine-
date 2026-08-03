"""
Phase 5: Password Hashing & JWT Security Service
------------------------------------------------
Uses SHA-256 + Salt hashing and PyJWT token encoding.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
import jwt
from app.core.config import settings

SALT = "custom_gpt_salt_token_2026"


def hash_password(password: str) -> str:
    """Hashes password using SHA-256 with salt."""
    salted = f"{SALT}{password}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies plain password against hashed password digest."""
    return hash_password(plain_password) == hashed_password


def create_access_token(data: Dict[str, str], expires_delta: Optional[timedelta] = None) -> str:
    """Generates signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Optional[Dict[str, str]]:
    """Decodes and validates JWT access token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None
