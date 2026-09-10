import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


from starlette.responses import JSONResponse


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_records = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limit for OPTIONS preflight requests
        if request.method == "OPTIONS":
            return await call_next(request)

        # Apply rate limiting only to API endpoints
        if request.url.path.startswith("/api/analyze"):
            client_ip = request.client.host if (request.client and hasattr(request.client, 'host')) else "127.0.0.1"
            now = time.time()
            
            # Clean old records
            timestamps = self.request_records[client_ip]
            valid_timestamps = [ts for ts in timestamps if now - ts < self.window_seconds]
            self.request_records[client_ip] = valid_timestamps
            
            if len(valid_timestamps) >= self.max_requests:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds} seconds."},
                    headers={"Access-Control-Allow-Origin": "*"}
                )
            
            self.request_records[client_ip].append(now)
            
        response = await call_next(request)
        return response
