"""
Phase 5: User & Database Model Store.
"""

from typing import Dict, Optional
from app.core.security import hash_password, verify_password


class InMemoryDatabase:
    """Mock Database handling User Registration & Session Auth."""

    def __init__(self):
        self.users: Dict[str, Dict[str, str]] = {}
        # Pre-seed demo admin user
        self.register_user("admin", "admin123")

    def register_user(self, username: str, password: str) -> bool:
        if username in self.users:
            return False
        self.users[username] = {
            "username": username,
            "password_hash": hash_password(password)
        }
        return True

    def authenticate_user(self, username: str, password: str) -> bool:
        user = self.users.get(username)
        if not user:
            return False
        return verify_password(password, user["password_hash"])


db = InMemoryDatabase()
