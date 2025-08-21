"""
Security service for handling brute force protection and security-related functionality.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer

logger = logging.getLogger("api.services.security_service")

class BruteForceProtection:
    """
    Brute force protection service to prevent multiple failed login attempts.
    """
    _instance = None
    _failed_attempts: Dict[str, list] = {}
    MAX_ATTEMPTS = 5
    LOCKOUT_TIME = 300  # 5 minutes in seconds

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BruteForceProtection, cls).__new__(cls)
        return cls._instance

    def is_blocked(self, key: str) -> bool:
        """Check if the key (IP or email) is blocked due to too many failed attempts."""
        if key not in self._failed_attempts:
            return False
        
        # Clean old attempts
        now = time.time()
        self._failed_attempts[key] = [t for t in self._failed_attempts[key] if now - t <= self.LOCKOUT_TIME]
        
        if len(self._failed_attempts[key]) >= self.MAX_ATTEMPTS:
            logger.warning(f"Brute force protection: Too many attempts for {key}")
            return True
        return False

    def register_attempt(self, key: str, success: bool):
        """Register a login attempt."""
        now = time.time()
        if key not in self._failed_attempts:
            self._failed_attempts[key] = []
        
        if success:
            # Reset failed attempts on successful login
            self._failed_attempts[key] = []
        else:
            self._failed_attempts[key].append(now)
            logger.warning(f"Failed login attempt for {key}")

    def get_remaining_time(self, key: str) -> int:
        """Get remaining lockout time in seconds."""
        if not self.is_blocked(key):
            return 0
        
        now = time.time()
        oldest_attempt = min(self._failed_attempts[key])
        return int((oldest_attempt + self.LOCKOUT_TIME) - now)

# Singleton instance
brute_force_protection = BruteForceProtection()

def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    if "x-forwarded-for" in request.headers:
        return request.headers["x-forwarded-for"].split(",")[0]
    return request.client.host if request.client else "unknown"

def check_brute_force(identifier: str):
    """Check if the identifier is blocked due to too many failed attempts."""
    if brute_force_protection.is_blocked(identifier):
        remaining = brute_force_protection.get_remaining_time(identifier)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Too many failed attempts",
                "retry_after_seconds": remaining
            },
            headers={"Retry-After": str(remaining)}
        )
