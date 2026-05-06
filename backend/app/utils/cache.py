"""
In-memory cache, rate limiting, and safe-API-call decorators.
"""
import time
import logging
from functools import wraps

from flask import request, jsonify

from app.config import CACHE_DURATION

logger = logging.getLogger(__name__)

# Shared in-memory stores
cache = {}
rate_limit_store = {}


def cache_response(timeout=CACHE_DURATION):
    """Simple in-memory cache decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"api:{f.__name__}:{hash(str(args) + str(kwargs))}"
            current_time = time.time()

            if cache_key in cache:
                cached_time, cached_result = cache[cache_key]
                if current_time - cached_time < timeout:
                    return cached_result

            result = f(*args, **kwargs)
            cache[cache_key] = (current_time, result)

            # Evict expired entries
            for key in list(cache.keys()):
                if current_time - cache[key][0] > timeout:
                    del cache[key]

            return result
        return decorated_function
    return decorator


def safe_api_call(f):
    """Decorator for safe API calls with logging."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            logger.info(f"API call: {f.__name__}")
            result = f(*args, **kwargs)
            logger.info(f"API call successful: {f.__name__}")
            return result
        except Exception as e:
            logger.error(f"API call failed: {f.__name__} - {str(e)}")
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500
    return decorated_function


def rate_limit(max_requests=60, window=60):
    """Simple in-memory rate limiting decorator."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            key = f"rate_limit:{client_ip}:{f.__name__}"
            current_time = time.time()

            # Evict expired entries
            for k in list(rate_limit_store.keys()):
                if current_time - rate_limit_store[k]['timestamp'] > window:
                    del rate_limit_store[k]

            if key in rate_limit_store:
                if rate_limit_store[key]['count'] >= max_requests:
                    return jsonify({"error": "Rate limit exceeded"}), 429
                rate_limit_store[key]['count'] += 1
            else:
                rate_limit_store[key] = {'count': 1, 'timestamp': current_time}

            return f(*args, **kwargs)
        return decorated_function
    return decorator
