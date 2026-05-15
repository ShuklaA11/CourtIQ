"""Game-clock math for lineup reconstruction.

The NBA play-by-play clock counts DOWN within each period: a regulation quarter
starts at 12:00 and an overtime period at 5:00, both ticking to 0:00. Every raw
action carries the *remaining* time in ISO-8601 duration form, e.g. "PT11M34.00S".

Reconstruction needs the opposite view — how much time has *elapsed* — in two
frames:
  * ``elapsed_in_period``: seconds since the current period tipped off.
  * ``cumulative_elapsed``: seconds since the opening tip of the game.

Centralizing this arithmetic means the state machine never re-derives period
lengths or re-parses clock strings, and every stint boundary is measured against
one consistent clock.
"""

from __future__ import annotations

import re

REGULATION_PERIOD_SECONDS = 12 * 60  # 720
OVERTIME_PERIOD_SECONDS = 5 * 60     # 300
REGULATION_PERIODS = 4

# "PT11M34.00S" -> minutes group optional (some clocks read "PT34.00S" under a
# minute); seconds may carry a fractional part.
_CLOCK_RE = re.compile(r"^PT(?:(\d+)M)?(\d+(?:\.\d+)?)S$")


def period_length_seconds(period: int) -> int:
    """Total length of ``period`` in seconds (regulation quarter vs. overtime)."""
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if period <= REGULATION_PERIODS:
        return REGULATION_PERIOD_SECONDS
    return OVERTIME_PERIOD_SECONDS


def period_start_cumulative(period: int) -> int:
    """Cumulative game seconds elapsed at the tip-off of ``period``.

    Period 1 tips at 0s; period 5 (first OT) tips at 4*720 = 2880s; and so on.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    full_regulation = min(period - 1, REGULATION_PERIODS)
    full_overtime = max(0, period - 1 - REGULATION_PERIODS)
    return (
        full_regulation * REGULATION_PERIOD_SECONDS
        + full_overtime * OVERTIME_PERIOD_SECONDS
    )


def parse_clock(clock: str) -> float:
    """Parse an ISO-8601 duration clock like "PT11M34.00S" to seconds remaining."""
    match = _CLOCK_RE.match(clock.strip())
    if not match:
        raise ValueError(f"unrecognized clock format: {clock!r}")
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def elapsed_in_period(period: int, clock: str) -> float:
    """Seconds elapsed since ``period`` tipped off, from the remaining-time clock."""
    remaining = parse_clock(clock)
    length = period_length_seconds(period)
    if remaining < 0 or remaining > length:
        raise ValueError(
            f"clock {clock!r} ({remaining}s remaining) out of range for period "
            f"{period} (length {length}s)"
        )
    return length - remaining


def cumulative_elapsed(period: int, clock: str) -> float:
    """Seconds elapsed since the opening tip: full prior periods + this period."""
    return period_start_cumulative(period) + elapsed_in_period(period, clock)
