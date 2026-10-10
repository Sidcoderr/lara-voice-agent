"""UK time and surgery opening hours.

Uses a small built-in rule for British Summer Time instead of a timezone
database, so it works on any machine with no extra packages (Windows has no
built-in timezone database). The surgery is in the UK even if you run the demo
somewhere else, so everything here is in UK local time.
"""

import datetime as dt

OPEN_TIME = dt.time(8, 0)
CLOSE_TIME = dt.time(18, 30)
HOURS_TEXT = "Monday to Friday, 8am to 6:30pm"
WEEKDAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]


def _last_sunday(year: int, month: int) -> dt.date:
    if month == 12:
        last_day = dt.date(year, 12, 31)
    else:
        last_day = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return last_day - dt.timedelta(days=(last_day.weekday() + 1) % 7)


def uk_now(now_utc: dt.datetime | None = None) -> dt.datetime:
    """Current UK wall-clock time as a naive datetime.

    British Summer Time runs from 01:00 UTC on the last Sunday of March to
    01:00 UTC on the last Sunday of October.
    """
    now_utc = now_utc or dt.datetime.now(dt.timezone.utc)
    year = now_utc.year
    bst_start = dt.datetime.combine(
        _last_sunday(year, 3), dt.time(1, 0), tzinfo=dt.timezone.utc
    )
    bst_end = dt.datetime.combine(
        _last_sunday(year, 10), dt.time(1, 0), tzinfo=dt.timezone.utc
    )
    offset_hours = 1 if bst_start <= now_utc < bst_end else 0
    return (now_utc + dt.timedelta(hours=offset_hours)).replace(tzinfo=None)


def uk_today(now: dt.datetime | None = None) -> dt.date:
    return (now or uk_now()).date()


# --- Speakable formatting ----------------------------------------------------


def format_time(value: dt.time | str) -> str:
    """'14:30' -> '2:30pm', '08:00' -> '8am', '12:00' -> '12pm'."""
    if isinstance(value, str):
        hour, minute = (int(part) for part in value.split(":")[:2])
    else:
        hour, minute = value.hour, value.minute
    suffix = "am" if hour < 12 else "pm"
    hour12 = hour % 12 or 12
    return f"{hour12}{f':{minute:02d}' if minute else ''}{suffix}"


def speak_date(day: dt.date | str) -> str:
    """'2026-10-06' -> 'Tuesday 6 October'."""
    if isinstance(day, str):
        day = dt.date.fromisoformat(day)
    return f"{WEEKDAY_NAMES[day.weekday()]} {day.day} {day.strftime('%B')}"


def speak_when(day: dt.date | str, time_text: str) -> str:
    return f"{speak_date(day)} at {format_time(time_text)}"


# --- Opening hours -------------------------------------------------------------


def is_open(now: dt.datetime | None = None) -> bool:
    now = now or uk_now()
    return now.weekday() < 5 and OPEN_TIME <= now.time() < CLOSE_TIME


def next_opening(now: dt.datetime | None = None) -> str:
    now = now or uk_now()
    if now.weekday() < 5 and now.time() < OPEN_TIME:
        return f"today at {format_time(OPEN_TIME)}"
    for days_ahead in range(1, 8):
        candidate = now.date() + dt.timedelta(days=days_ahead)
        if candidate.weekday() < 5:
            name = "tomorrow" if days_ahead == 1 else WEEKDAY_NAMES[candidate.weekday()]
            return f"{name} at {format_time(OPEN_TIME)}"
    return f"Monday at {format_time(OPEN_TIME)}"


def surgery_status(now: dt.datetime | None = None) -> dict:
    """Live open/closed status, with advice if the surgery is closed."""
    now = now or uk_now()
    status = {
        "day": WEEKDAY_NAMES[now.weekday()],
        "time_now": format_time(now.time().replace(second=0, microsecond=0)),
        "opening_hours": HOURS_TEXT,
    }
    if is_open(now):
        status["open_now"] = True
        status["message"] = f"The surgery is open until {format_time(CLOSE_TIME)} today."
    else:
        status["open_now"] = False
        status["next_opening"] = next_opening(now)
        status["message"] = (
            "The surgery is closed right now. Appointments can still be booked and "
            "messages taken. For urgent medical advice call NHS 111; in an emergency "
            "call 999."
        )
    return status


def day_hours(day: dt.date) -> dict:
    """Opening hours for a specific date (the surgery is closed at weekends)."""
    if day.weekday() < 5:
        return {
            "date": speak_date(day),
            "open": True,
            "hours": f"{format_time(OPEN_TIME)} to {format_time(CLOSE_TIME)}",
        }
    return {
        "date": speak_date(day),
        "open": False,
        "message": "The surgery is closed at weekends. For urgent advice call NHS 111.",
    }


def resolve_day(text: str, today: dt.date | None = None) -> dt.date | None:
    """Understand 'today', 'tomorrow', a weekday name, or an ISO date."""
    today = today or uk_today()
    cleaned = (text or "").strip().lower()
    if cleaned in ("", "today", "now"):
        return today
    if cleaned == "tomorrow":
        return today + dt.timedelta(days=1)
    for index, name in enumerate(WEEKDAY_NAMES):
        if cleaned in (name.lower(), name.lower()[:3]):
            return today + dt.timedelta(days=(index - today.weekday()) % 7)
    try:
        return dt.date.fromisoformat(cleaned)
    except ValueError:
        return None
