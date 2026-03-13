"""
middleware/__init__.py
======================
Middleware components for AI Lawyer backend.
"""
from middleware.rate_limiter import RateLimiterMiddleware

__all__ = ["RateLimiterMiddleware"]
