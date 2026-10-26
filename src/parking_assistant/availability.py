from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

Interval = tuple[datetime, datetime]


def overlaps(a: Interval, b: Interval) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def peak_occupancy(bookings: Iterable[Interval], window: Interval) -> int:
    win_start, win_end = window
    events: list[tuple[datetime, int]] = []
    for start, end in bookings:
        if not overlaps((start, end), window):
            continue
        events.append((max(start, win_start), 1))
        events.append((min(end, win_end), -1))
    events.sort(key=lambda event: (event[0], event[1]))

    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def free_spaces(total: int, bookings: Iterable[Interval], window: Interval) -> int:
    return max(total - peak_occupancy(bookings, window), 0)
