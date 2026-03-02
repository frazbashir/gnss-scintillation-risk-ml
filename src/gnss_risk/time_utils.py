from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, List

UTC = timezone.utc
UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(UTC)


def to_utc_string(value: datetime) -> str:
    return value.astimezone(UTC).strftime(UTC_FMT)


def build_timeline(start: datetime, end: datetime, cadence_minutes: int) -> List[datetime]:
    step = timedelta(minutes=cadence_minutes)
    points: List[datetime] = []
    cursor = start
    while cursor <= end:
        points.append(cursor)
        cursor += step
    return points


def shift_minutes(value: datetime, minutes: int) -> datetime:
    return value + timedelta(minutes=minutes)
