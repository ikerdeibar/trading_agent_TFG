# src/core/clock.py
from __future__ import annotations
from datetime import datetime, date, timedelta, time
from zoneinfo import ZoneInfo
import exchange_calendars as xcals

_ET  = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# NYSE calendar
_NYSE = xcals.get_calendar("XNYS")

# Actionable window matches judge rule_market_hours
_WINDOW_OPEN  = time(9, 45)
_WINDOW_CLOSE = time(15, 45)


# ── Market status ─────────────────────────────────────────────────────────────

def is_trading_day(d: date | None = None) -> bool:
    """Return True if d (default: today ET) is a NYSE trading day."""
    if d is None:
        d = datetime.now(tz=_ET).date()
    return _NYSE.is_session(d.isoformat())


def is_market_open(now: datetime | None = None) -> bool:
    """
    Return True if now falls within the actionable window (09:45–15:45 ET)
    on a NYSE trading day.
    """
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET)

    if not is_trading_day(now.date()):
        return False
    return _WINDOW_OPEN <= now.time() <= _WINDOW_CLOSE


def next_trading_day(from_date: date | None = None) -> date:
    """Return the next NYSE trading day after from_date (default: today ET)."""
    if from_date is None:
        from_date = datetime.now(tz=_ET).date()
    candidate = from_date + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def minutes_to_close(now: datetime | None = None) -> float:
    """Return minutes remaining until 15:45 ET. Negative if after close."""
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET)
    close_dt = now.replace(
        hour=_WINDOW_CLOSE.hour,
        minute=_WINDOW_CLOSE.minute,
        second=0,
        microsecond=0,
    )
    return (close_dt - now).total_seconds() / 60


# ── Cycle scheduling ──────────────────────────────────────────────────────────

def build_cycle_schedule(
    session_date:            date,
    cycle_interval_minutes:  int = 65,
    num_cycles:              int = 6,
    session_start_et:        str = "09:45",
) -> list[datetime]:
    """
    Build the list of N cycle start times for a session day.
    All times are timezone-aware (ET).

    Example with defaults (65-min cadence, 6 cycles):
      09:45, 10:50, 11:55, 13:00, 14:05, 15:10
    Last cycle must start by 15:10 to complete before 15:45 close.
    """
    start_h, start_m = map(int, session_start_et.split(":"))
    base = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        start_h,
        start_m,
        tzinfo=_ET,
    )
    return [
        base + timedelta(minutes=i * cycle_interval_minutes)
        for i in range(num_cycles)
    ]


def current_cycle_index(
    schedule:   list[datetime],
    now:        datetime | None = None,
) -> int | None:
    """
    Return the index (0-based) of the cycle that should be running now.
    Returns None if now is before the first cycle or after the last.
    """
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET)

    active = None
    for i, cycle_start in enumerate(schedule):
        if now >= cycle_start:
            active = i
    return active


def should_abort_cycle(
    cycle_start: datetime,
    max_duration_minutes: int = 55,
    now: datetime | None = None,
) -> bool:
    """
    Return True if a cycle has been running longer than max_duration_minutes.
    Prevents a slow LLM call from bleeding into the next cycle window.
    Default: abort if cycle exceeds 55 minutes (leaves 10-min buffer before
    next 65-min cycle starts).
    """
    if now is None:
        now = datetime.now(tz=_ET)
    else:
        now = now.astimezone(_ET)

    elapsed = (now - cycle_start).total_seconds() / 60
    return elapsed > max_duration_minutes


def et_now() -> datetime:
    """Return current time as timezone-aware ET datetime."""
    return datetime.now(tz=_ET)


def utc_now() -> datetime:
    """Return current time as timezone-aware UTC datetime."""
    return datetime.now(tz=_UTC)
