"""
FastAPI middleware: request logging, timing, and error handling.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

log = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with timing information."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        log.info(
            f"[{request_id}] ← {request.method} {request.url.path}"
        )

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
            log.info(
                f"[{request_id}] → {response.status_code} ({elapsed_ms:.1f}ms)"
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.error(f"[{request_id}] ✖ {exc} ({elapsed_ms:.1f}ms)")
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )
