from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from parking_assistant.facility import TIMEZONE_NAME

PARKING_TZ = ZoneInfo(TIMEZONE_NAME)

Clock = Callable[[], datetime]

_DAY_WORDS = {
    "today": 0,
    "tonight": 0,
    "tomorrow": 1,
    "tomorow": 1,
    "tommorow": 1,
    "tommorrow": 1,
    "tomorrrow": 1,
    "tmrw": 1,
    "tmr": 1,
}
_DAY_ALT = "|".join(sorted(_DAY_WORDS, key=len, reverse=True))
_TIME = r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?"
_TIME_ONLY = re.compile(rf"^(?:at\s+)?{_TIME}$", re.IGNORECASE)
_DAY_THEN_TIME = re.compile(rf"^({_DAY_ALT})\s+(?:at\s+)?{_TIME}$", re.IGNORECASE)
_TIME_THEN_DAY = re.compile(rf"^(?:at\s+)?{_TIME}\s+({_DAY_ALT})$", re.IGNORECASE)
_LOCAL_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M")
EXAMPLE = "2026-11-10 09:00 or tomorrow 9:00"


class DateParseError(ValueError):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


def _clock_time(hour: str, minute: str | None, meridiem: str | None) -> tuple[int, int]:
    h, m = int(hour), int(minute or 0)
    if meridiem:
        if not 1 <= h <= 12:
            raise DateParseError("That time doesn't look right.")
        pm = meridiem.lower().startswith("p")
        h = (h % 12) + (12 if pm else 0)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise DateParseError("That time doesn't look right.")
    return h, m


def _local(day, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=PARKING_TZ)


def parse_user_datetime(
    text: str, now: datetime | None = None, base: datetime | None = None
) -> datetime:
    value = " ".join(text.strip().split())
    if not value:
        raise DateParseError("Please tell me a date and time.")
    now = now or utc_now()
    today = now.astimezone(PARKING_TZ).date()

    match = _DAY_THEN_TIME.match(value) or _TIME_THEN_DAY.match(value)
    if match:
        groups = match.groups()
        if match.re is _DAY_THEN_TIME:
            word, hour, minute, meridiem = groups
        else:
            hour, minute, meridiem, word = groups
        h, m = _clock_time(hour, minute, meridiem)
        day = today + timedelta(days=_DAY_WORDS[word.lower()])
        return _local(day, h, m).astimezone(UTC)

    match = _TIME_ONLY.match(value)
    if match and (match.group(2) or match.group(3) or base is not None):
        h, m = _clock_time(*match.groups())
        if base is not None:
            local_base = base.astimezone(PARKING_TZ)
            result = _local(local_base.date(), h, m)
            if result <= local_base:
                result += timedelta(days=1)
            return result.astimezone(UTC)
        result = _local(today, h, m)
        if result.astimezone(UTC) <= now:
            result += timedelta(days=1)
        return result.astimezone(UTC)

    cleaned = re.sub(r"\s+at\s+", " ", value, flags=re.IGNORECASE)
    iso = cleaned.replace("Z", "+00:00") if cleaned.endswith("Z") else cleaned
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
    if parsed is not None:
        if len(iso) <= 10:
            raise DateParseError(
                "Got the date, but what time? For example 2026-11-10 09:00."
            )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=PARKING_TZ)
        return parsed.astimezone(UTC)

    for fmt in _LOCAL_FORMATS:
        try:
            local = datetime.strptime(cleaned, fmt).replace(tzinfo=PARKING_TZ)
        except ValueError:
            continue
        return local.astimezone(UTC)

    raise DateParseError(
        f"Sorry, I couldn't read \u201c{value}\u201d as a date and time. "
        f"Try something like {EXAMPLE}."
    )


def to_utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("naive datetime")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def from_utc_text(text: str) -> datetime:
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        raise ValueError("stored datetime without offset")
    return value.astimezone(UTC)


def to_local(value: datetime) -> datetime:
    return value.astimezone(PARKING_TZ)


def local_iso(value: datetime) -> str:
    return to_local(value).isoformat(timespec="seconds")


def human_local(value: datetime) -> str:
    return to_local(value).strftime("%Y-%m-%d %H:%M") + f" ({TIMEZONE_NAME})"


def friendly_local(value: datetime) -> str:
    local = to_local(value)
    return f"{local:%a} {local.day} {local:%b %Y}, {local:%H:%M}"
