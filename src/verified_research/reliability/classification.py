"""Failure classification and secret sanitization for the reliability layer."""

import json
import re
import socket
from typing import Any
from pydantic import ValidationError
from verified_research.reliability.models import ErrorCategory

# Compiled regex patterns for redacting sensitive secrets and credentials
SECRET_PATTERNS = [
    (re.compile(r"gsk_[a-zA-Z0-9_\-]{16,}", re.IGNORECASE), "[REDACTED_GROQ_KEY]"),
    (re.compile(r"tvly-[a-zA-Z0-9_\-]{16,}", re.IGNORECASE), "[REDACTED_TAVILY_KEY]"),
    (re.compile(r"sk-[a-zA-Z0-9_\-]{16,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{16,}", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
    (
        re.compile(
            r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{8,})['\"]?",
            re.IGNORECASE,
        ),
        r"\1=[REDACTED]",
    ),
]


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages to ensure API keys, tokens, and credentials are never leaked.

    Args:
        message: Raw error message string.

    Returns:
        Sanitized message string with sensitive credentials replaced by redaction placeholders.
    """
    sanitized = str(message)
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def extract_status_code(exc: Exception) -> int | None:
    """Extract an HTTP status code from an exception if present.

    Inspects common HTTP client attributes (.status_code, .response.status_code, .code)
    and falls back to regular expression inspection of error text.

    Args:
        exc: Caught exception.

    Returns:
        Integer status code if resolved, otherwise None.
    """
    # 1. Attribute inspection
    if hasattr(exc, "status_code") and isinstance(exc.status_code, int):
        return exc.status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        if isinstance(exc.response.status_code, int):
            return exc.response.status_code
    if hasattr(exc, "code") and isinstance(exc.code, int):
        return exc.code

    # 2. String search for standard HTTP status codes
    msg = str(exc)
    match = re.search(r"\b(400|401|403|404|408|422|429|500|502|503|504)\b", msg)
    if match:
        return int(match.group(1))

    return None


def classify_error(exc: Exception) -> tuple[ErrorCategory, bool, int | None]:
    """Classify an operation failure into an ErrorCategory and determine retryability.

    Decision Matrix:
        - Network Timeout (408, TimeoutError, socket.timeout) -> RETRYABLE
        - Rate Limit (429, 'rate limit', 'too many requests') -> RETRYABLE
        - Server Error (500, 502, 503, 504, ConnectionError)  -> RETRYABLE
        - Malformed Output (ValidationError, JSONDecodeError) -> RETRYABLE (bounded)
        - Client Error (400, 404, 422, 'bad request')         -> NON-RETRYABLE
        - Auth / Credential (401, 403, MissingApiKeyError)     -> NON-RETRYABLE
        - Configuration (unsupported provider, ValueError)   -> NON-RETRYABLE
        - Unknown                                             -> NON-RETRYABLE

    Args:
        exc: The caught exception.

    Returns:
        Tuple of (ErrorCategory, is_retryable, status_code).
    """
    status_code = extract_status_code(exc)
    msg = str(exc).lower()
    type_name = type(exc).__name__.lower()

    # 1. Timeout / Network Timeout (Retryable)
    if (
        status_code == 408
        or isinstance(exc, (TimeoutError, socket.timeout))
        or "timeout" in msg
        or "timed out" in msg
        or "timeout" in type_name
    ):
        return ErrorCategory.NETWORK_TIMEOUT, True, status_code or 408

    # 2. Rate Limit (Retryable)
    if (
        status_code == 429
        or "rate limit" in msg
        or "ratelimit" in msg
        or "too many requests" in msg
        or "429" in msg
        or "ratelimit" in type_name
    ):
        return ErrorCategory.RATE_LIMIT, True, 429

    # 3. Server Errors & Connection Drops (Retryable)
    if (
        status_code in (500, 502, 503, 504)
        or isinstance(exc, (ConnectionError, BrokenPipeError, ConnectionResetError))
        or "connection" in type_name
        or "500 internal server error" in msg
        or "bad gateway" in msg
        or "service unavailable" in msg
        or "gateway timeout" in msg
    ):
        return ErrorCategory.SERVER_ERROR, True, status_code or 500

    # 4. Authentication / Permission Errors (Non-Retryable: Fail Fast)
    if (
        status_code in (401, 403)
        or "missingapikeyerror" in type_name
        or "authentication" in type_name
        or "unauthorized" in msg
        or "forbidden" in msg
        or "invalid api key" in msg
        or "api key is missing" in msg
        or "authentication failed" in msg
    ):
        return ErrorCategory.AUTH_ERROR, False, status_code or 401

    # 5. Client Request Errors (Non-Retryable: Fail Fast)
    if (
        status_code in (400, 404, 422)
        or "bad request" in msg
        or "not found" in msg
        or "unprocessable" in msg
    ):
        return ErrorCategory.CLIENT_ERROR, False, status_code or 400

    # 6. Malformed Structured Output from LLM (Bounded Retryable)
    if (
        isinstance(exc, (ValidationError, json.JSONDecodeError))
        or "validation error" in msg
        or "validationerror" in type_name
        or "outputparser" in type_name
        or "jsondecodeerror" in type_name
    ):
        return ErrorCategory.MALFORMED_OUTPUT, True, None

    # 7. Configuration Errors (Non-Retryable: Fail Fast)
    if (
        isinstance(exc, (ValueError, KeyError))
        or "configuration" in msg
        or "unsupported" in msg
    ):
        return ErrorCategory.CONFIGURATION_ERROR, False, None

    # 8. Unclassified Failure (Safe Default: Non-Retryable)
    return ErrorCategory.UNKNOWN, False, status_code
