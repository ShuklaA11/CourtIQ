"""Polite, resilient wrapper around nba_api endpoint calls.

stats.nba.com rate-limits aggressively and times out often. Every network call
in the ingest goes through `polite_call`, which enforces a minimum gap between
requests and retries transient failures with exponential backoff + jitter.
"""

from __future__ import annotations

import json
import random
import time
from typing import Callable, TypeVar

import requests

T = TypeVar("T")

# Only these are worth retrying: the same call might succeed moments later.
# A timeout/connection drop is transient; a rate-limited response comes back as
# non-JSON (HTML error page), surfacing as a JSONDecodeError. Anything else
# (KeyError, TypeError, ...) is a deterministic bug — retrying only hides it.
RETRYABLE = (requests.exceptions.RequestException, json.JSONDecodeError)

# --- Tuning knobs (see teaching note in README on why these values) ---
MIN_GAP_SECONDS = 0.75      # floor on time between successive requests
MAX_RETRIES = 8             # attempts before giving up on a single call
BASE_BACKOFF_SECONDS = 1.5  # first retry waits ~this long, then doubles
BACKOFF_CAP_SECONDS = 60.0  # never wait longer than this between retries
REQUEST_TIMEOUT = 60        # per-request timeout handed to nba_api

# Wall-clock timestamp of the last request, module-level so the gap is enforced
# across every call site, not per-caller.
_last_request_at = 0.0


def _respect_min_gap() -> None:
    """Sleep just long enough that requests are spaced >= MIN_GAP_SECONDS apart."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_GAP_SECONDS:
        time.sleep(MIN_GAP_SECONDS - elapsed)
    _last_request_at = time.monotonic()


def polite_call(make_endpoint: Callable[..., T], **kwargs) -> T:
    """Call an nba_api endpoint factory with pacing and retry.

    `make_endpoint` is the endpoint class (e.g. PlayByPlayV2); kwargs are its
    constructor args. We inject a timeout unless the caller set one.

    Retries any exception (timeouts, connection resets, 429s surface as these in
    nba_api) up to MAX_RETRIES, waiting an exponentially growing, jittered delay
    between attempts. Re-raises the last exception if all attempts fail.
    """
    kwargs.setdefault("timeout", REQUEST_TIMEOUT)
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        _respect_min_gap()
        try:
            return make_endpoint(**kwargs)
        except RETRYABLE as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                break
            # Exponential backoff: 1.5, 3, 6, 12, ... capped.
            backoff = min(BASE_BACKOFF_SECONDS * (2 ** attempt), BACKOFF_CAP_SECONDS)
            # Full jitter: sleep a random amount in [0, backoff] rather than the
            # exact backoff, so concurrent/repeated failures don't resynchronize.
            sleep_for = random.uniform(0, backoff)
            print(f"    retry {attempt + 1}/{MAX_RETRIES} after error "
                  f"({type(exc).__name__}); sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)

    raise RuntimeError(f"polite_call failed after {MAX_RETRIES} attempts") from last_exc
