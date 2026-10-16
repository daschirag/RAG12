"""Cross-cutting API behaviour: one JSON error format, request ids and a structured request log.

Every error response has the same body::

    {"code": "not_found", "message": "...", "hint": "...", "detail": <original detail>, "request_id": "..."}

``detail`` keeps FastAPI's original field (a string, or the validation error list for 422) so existing
clients keep working. Every response carries ``X-Request-ID`` (taken from the request if it is a safe token,
otherwise generated) and one JSON log line with method, path, status and latency.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

access_log = logging.getLogger("greenrag.access")
error_log = logging.getLogger("greenrag.error")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
         409: "conflict", 422: "invalid_request", 429: "rate_limited", 500: "internal_error",
         502: "upstream_error", 503: "service_unavailable", 504: "timeout"}
HINTS = {
    400: "Check the request parameters against the OpenAPI docs at /docs.",
    404: "Check the id or path; list endpoints such as /domains or /diagnose/events show valid values.",
    405: "Check the HTTP method in the OpenAPI docs at /docs.",
    422: "Fix the fields listed in 'detail' (see the request schema and examples at /docs).",
    500: "Unexpected server error. Quote the request_id; the server log has the stack trace.",
    503: "A model or result file is missing or still loading. Call POST /warmup and check GET /ready.",
    504: "The local LLM did not answer in time. Retry, or use the evidence-only response.",
}


class ApiError(HTTPException):
    """HTTPException with an explicit machine code and hint."""

    def __init__(self, status_code: int, message: str, code: str | None = None, hint: str | None = None) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code, self.hint = code, hint


def error_body(status: int, message: str, request: Request, detail=None, code: str | None = None,
               hint: str | None = None) -> dict:
    return {"code": code or CODES.get(status, "error"), "message": message,
            "hint": hint or HINTS.get(status, "See the OpenAPI docs at /docs."),
            "detail": message if detail is None else detail,
            "request_id": getattr(request.state, "request_id", None)}


def _validation_message(errors: list[dict]) -> str:
    parts = []
    for e in errors[:5]:
        loc = ".".join(str(x) for x in e.get("loc", ()) if x not in ("body", "query", "path"))
        parts.append(f"{loc or 'request'}: {e.get('msg', 'invalid')}")
    more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
    return "Request validation failed - " + "; ".join(parts) + more


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        rid = rid if _SAFE_ID.match(rid) else uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            access_log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method,
                                        "path": request.url.path, "status": status,
                                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}))
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        message = detail if isinstance(detail, str) else json.dumps(jsonable_encoder(detail))
        body = error_body(exc.status_code, message, request, detail=detail,
                          code=getattr(exc, "code", None), hint=getattr(exc, "hint", None))
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = jsonable_encoder(exc.errors())
        for e in errors:                       # never echo large or sensitive inputs back
            e.pop("input", None)
            e.pop("ctx", None)
        return JSONResponse(error_body(422, _validation_message(errors), request, detail=errors), status_code=422)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        error_log.exception("unhandled error request_id=%s path=%s", getattr(request.state, "request_id", None),
                            request.url.path)
        return JSONResponse(error_body(500, f"Internal error ({type(exc).__name__}).", request), status_code=500)
