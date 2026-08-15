"""Shared HTTP layer for the AI provider APIs.

Handles connection reuse, cancellation and — the main point of this module — reacting to the
rate-limit rejections each provider returns instead of trying to stay under a guessed
requests-per-minute ceiling. Callers just await a final response; the throttling is invisible
to them.
"""

import json
import logging
import random
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests  # type: ignore
from requests.adapters import HTTPAdapter  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_RETRY_WAIT_SECONDS = 120
# Fallback wait when a provider says "rate limited" without saying for how long
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60.0
# Cooldowns are slept off in slices this long so cancellation stays responsive
CANCEL_POLL_INTERVAL = 0.5

GEMINI = "gemini"
OPENAI = "openai"
ANTHROPIC = "anthropic"
TOGETHER = "together"


# --- Response classification -------------------------------------------------------------


class ResponseAction:
    """What post_with_retry should do with a response."""

    OK = "ok"
    RETRY = "retry"
    # Terminal: retrying cannot help (bad request, no credit, daily quota used up)
    FAIL = "fail"


def _error_body(response: "requests.Response") -> dict:
    """Parse the JSON error body, returning an empty dict if it isn't JSON."""
    try:
        decoded = json.loads(response.text)
    except (ValueError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def parse_seconds_header(value: Optional[str]) -> Optional[float]:
    """Parse a Retry-After header value expressed in seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None


def parse_rfc3339_reset(value: Optional[str]) -> Optional[float]:
    """Parse an RFC 3339 timestamp (Anthropic's *-reset headers) into seconds from now."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        reset_at = datetime.fromisoformat(text)
    except ValueError:
        return None
    if reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    return max(0.0, (reset_at - datetime.now(timezone.utc)).total_seconds())


_GO_DURATION_PART = re.compile(r"([0-9]*\.?[0-9]+)(ms|us|µs|ns|[smh])")
_GO_DURATION_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def parse_go_duration(value: Optional[str]) -> Optional[float]:
    """Parse OpenAI's Go-style duration strings, e.g. "1s", "6m0s", "20ms"."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parts = _GO_DURATION_PART.findall(text)
    if not parts:
        return None
    # Reject strings with junk outside the matched parts, e.g. "6m0s extra"
    if "".join(f"{amount}{unit}" for amount, unit in parts) != text:
        return None
    total = 0.0
    for amount, unit in parts:
        total += float(amount) * _GO_DURATION_UNITS[unit]
    return max(0.0, total)


# Google returns retry hints as protobuf-JSON durations, always plain seconds like "34s"
_GOOGLE_DURATION = re.compile(r"^([0-9]*\.?[0-9]+)s$")


def parse_google_duration(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = _GOOGLE_DURATION.match(value.strip())
    if not match:
        return None
    return max(0.0, float(match.group(1)))


def _gemini_details(body: dict) -> list:
    details = body.get("error", {}).get("details")
    return details if isinstance(details, list) else []


def _gemini_is_daily_quota(body: dict) -> bool:
    """A per-day quota won't clear during this run, so retrying is pointless."""
    for detail in _gemini_details(body):
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations")
        if not isinstance(violations, list):
            continue
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            quota_id = str(violation.get("quotaId", ""))
            if "PerDay" in quota_id:
                return True
    return False


def _gemini_retry_delay(body: dict) -> Optional[float]:
    for detail in _gemini_details(body):
        if not isinstance(detail, dict):
            continue
        if str(detail.get("@type", "")).endswith("google.rpc.RetryInfo"):
            delay = parse_google_duration(detail.get("retryDelay"))
            if delay is not None:
                return delay
    return None


def classify_response(provider: str, response: "requests.Response") -> tuple[str, Optional[float]]:
    """Decide what to do with a provider response.

    Returns (action, suggested_delay_seconds). The delay is None when the provider gave no
    hint; the caller then falls back to exponential backoff.
    """
    status = response.status_code
    if status == 200:
        return ResponseAction.OK, None

    headers = response.headers
    body = _error_body(response)

    if provider == ANTHROPIC:
        # 429 rate_limit_error, 529 overloaded_error, 500 api_error, 504 timeout_error
        if status in (429, 500, 502, 503, 504, 529):
            delay = parse_seconds_header(headers.get("retry-after"))
            if delay is None:
                # Fall back to whichever limit bucket resets soonest
                resets = [
                    parse_rfc3339_reset(headers.get(name))
                    for name in (
                        "anthropic-ratelimit-requests-reset",
                        "anthropic-ratelimit-input-tokens-reset",
                        "anthropic-ratelimit-output-tokens-reset",
                        "anthropic-ratelimit-tokens-reset",
                    )
                ]
                valid = [r for r in resets if r is not None]
                delay = min(valid) if valid else None
            return ResponseAction.RETRY, delay
        return ResponseAction.FAIL, None

    if provider == GEMINI:
        if status == 429:
            if _gemini_is_daily_quota(body):
                logger.error(
                    "Gemini daily quota exhausted, not retrying: %s",
                    body.get("error", {}).get("message", response.text),
                )
                return ResponseAction.FAIL, None
            return ResponseAction.RETRY, _gemini_retry_delay(body)
        if status in (500, 502, 503, 504):
            return ResponseAction.RETRY, _gemini_retry_delay(body)
        return ResponseAction.FAIL, None

    # OpenAI and Together share the same error shape
    if status == 429:
        code = str(body.get("error", {}).get("code") or body.get("error", {}).get("type") or "")
        if code == "insufficient_quota":
            logger.error(
                "OpenAI-compatible API reports insufficient quota (billing), not retrying: %s",
                body.get("error", {}).get("message", response.text),
            )
            return ResponseAction.FAIL, None
        delay = parse_seconds_header(headers.get("Retry-After"))
        if delay is None:
            resets = [
                parse_go_duration(headers.get(name))
                for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens")
            ]
            valid = [r for r in resets if r is not None]
            delay = max(valid) if valid else None
        return ResponseAction.RETRY, delay
    if status in (500, 502, 503, 504):
        return ResponseAction.RETRY, parse_seconds_header(headers.get("Retry-After"))
    return ResponseAction.FAIL, None


def preemptive_cooldown(provider: str, response: "requests.Response") -> Optional[float]:
    """On a successful response, check whether we just used up the last of a limit bucket.

    Returns how long to hold off before the next request to this model, or None when there is
    still headroom. Turns most would-be rejections into a wait.
    """
    headers = response.headers
    try:
        if provider == ANTHROPIC:
            remaining = headers.get("anthropic-ratelimit-requests-remaining")
            if remaining is not None and int(remaining) <= 0:
                return parse_rfc3339_reset(headers.get("anthropic-ratelimit-requests-reset"))
        elif provider in (OPENAI, TOGETHER):
            remaining = headers.get("x-ratelimit-remaining-requests")
            if remaining is not None and int(remaining) <= 0:
                return parse_go_duration(headers.get("x-ratelimit-reset-requests"))
    except (TypeError, ValueError):
        return None
    # Gemini does not return remaining-quota headers
    return None


# --- Rate limit tracking -----------------------------------------------------------------


class RateLimitTracker:
    """Tracks which models are currently in a rate-limit cooldown.

    Read and written from worker threads, so everything is under a lock. Keyed per
    provider+model because limits are enforced separately per model.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cooldown_until: dict[str, float] = {}
        self._consecutive: dict[str, int] = {}

    def wait_time(self, key: str) -> float:
        """Seconds left on this model's cooldown, 0 if it's clear."""
        with self._lock:
            until = self._cooldown_until.get(key)
            if until is None:
                return 0.0
            remaining = until - time.monotonic()
            if remaining <= 0:
                del self._cooldown_until[key]
                return 0.0
            return remaining

    def note_rate_limited(self, key: str, delay: float) -> None:
        """Hold off further requests to this model for `delay` seconds."""
        with self._lock:
            until = time.monotonic() + delay
            # Never shorten an existing cooldown
            self._cooldown_until[key] = max(until, self._cooldown_until.get(key, 0.0))
            self._consecutive[key] = self._consecutive.get(key, 0) + 1

    def note_success(self, key: str) -> None:
        with self._lock:
            self._cooldown_until.pop(key, None)
            self._consecutive.pop(key, None)

    def consecutive_failures(self, key: str) -> int:
        with self._lock:
            return self._consecutive.get(key, 0)

    def reset(self) -> None:
        with self._lock:
            self._cooldown_until.clear()
            self._consecutive.clear()


rate_limit_tracker = RateLimitTracker()


# --- Sessions ----------------------------------------------------------------------------

_session_lock = threading.Lock()
_sessions: dict[str, "requests.Session"] = {}
_pool_size = 32


def set_connection_pool_size(size: int) -> None:
    """Size the connection pools to the concurrency ceiling.

    Called at the start of a bulk run. Existing sessions are dropped so the new size takes
    effect; in-flight requests keep their own connection alive until they finish.
    """
    global _pool_size
    with _session_lock:
        if size == _pool_size:
            return
        _pool_size = max(1, size)
        _sessions.clear()


def get_session(provider: str) -> "requests.Session":
    """One connection-pooled session per provider, created lazily."""
    with _session_lock:
        session = _sessions.get(provider)
        if session is None:
            session = requests.Session()
            # max_retries=0: retries are handled by post_with_retry, which knows how to read
            # each provider's rate-limit hints
            adapter = HTTPAdapter(
                pool_connections=_pool_size,
                pool_maxsize=_pool_size,
                max_retries=0,
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            _sessions[provider] = session
        return session


def close_all_sessions() -> None:
    """Close every session, aborting requests in flight.

    Used to make cancellation take effect immediately. Sessions are re-created lazily, so this
    is safe to call at any point.
    """
    with _session_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        try:
            session.close()
        except Exception as e:
            logger.debug("Error closing session: %s", e)


# --- The request loop --------------------------------------------------------------------


def _sleep_cancellable(seconds: float, cancel_state: Optional[Any]) -> bool:
    """Sleep in short slices so cancellation isn't delayed. Returns False if cancelled."""
    deadline = time.monotonic() + seconds
    while True:
        if cancel_state is not None and cancel_state.is_cancelled():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(CANCEL_POLL_INTERVAL, remaining))


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter, for when the provider gave no hint."""
    return min(DEFAULT_RATE_LIMIT_WAIT_SECONDS, float(2**attempt)) + random.uniform(0, 1)


def post_with_retry(
    provider: str,
    model: str,
    url: str,
    headers: dict,
    json_body: dict,
    timeout: float,
    cancel_state: Optional[Any] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_retry_wait: float = DEFAULT_MAX_RETRY_WAIT_SECONDS,
) -> Optional["requests.Response"]:
    """POST to a provider, waiting out rate limits and retrying transient failures.

    Blocking; expected to be called from a worker thread. Returns the final response — which
    may be a non-200 the caller should log — or None if the request was cancelled or every
    attempt failed to produce a response.
    """
    key = f"{provider}:{model}"
    session = get_session(provider)
    last_response: Optional["requests.Response"] = None

    for attempt in range(max_retries + 1):
        if cancel_state is not None and cancel_state.is_cancelled():
            return None

        # Another task may already have been rejected for this model; wait rather than
        # spending a request on the same rejection.
        cooldown = rate_limit_tracker.wait_time(key)
        if cooldown > 0:
            logger.debug("Waiting %.1fs on active cooldown for %s", cooldown, key)
            if not _sleep_cancellable(cooldown, cancel_state):
                return None

        try:
            response = session.post(url, headers=headers, json=json_body, timeout=timeout)
        except requests.exceptions.Timeout:
            logger.warning("Request to %s timed out (attempt %d)", key, attempt + 1)
            action, delay = ResponseAction.RETRY, None
            response = None
        except requests.exceptions.RequestException as e:
            # Includes connection errors, and the aborted-socket error raised when the
            # session is closed by a cancellation
            if cancel_state is not None and cancel_state.is_cancelled():
                return None
            logger.warning("Request to %s failed (attempt %d): %s", key, attempt + 1, e)
            action, delay = ResponseAction.RETRY, None
            response = None
        else:
            last_response = response
            action, delay = classify_response(provider, response)

        if action == ResponseAction.OK and response is not None:
            rate_limit_tracker.note_success(key)
            hold_off = preemptive_cooldown(provider, response)
            if hold_off:
                logger.debug("Model %s is out of request quota, holding off %.1fs", key, hold_off)
                rate_limit_tracker.note_rate_limited(key, hold_off)
            return response

        if action == ResponseAction.FAIL:
            return last_response

        # Retryable
        if attempt >= max_retries:
            logger.error("Giving up on %s after %d attempts", key, attempt + 1)
            return last_response

        if delay is None:
            delay = _backoff_delay(attempt)
        if delay > max_retry_wait:
            logger.error(
                "%s asked to wait %.0fs, above max_retry_wait_seconds (%.0fs) — giving up",
                key,
                delay,
                max_retry_wait,
            )
            return last_response

        status = response.status_code if response is not None else "no response"
        logger.warning("Retrying %s in %.1fs (status %s)", key, delay, status)
        rate_limit_tracker.note_rate_limited(key, delay)
        if not _sleep_cancellable(delay, cancel_state):
            return None

    return last_response
