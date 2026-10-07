import json
import re
from datetime import date
from pathlib import Path

from errors import ValidationError

DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse_date(value, name):
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not DATE.fullmatch(value):
        raise ValidationError(f"'{name}' must be a date in YYYY-MM-DD format")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValidationError(f"'{name}' is not a valid calendar date") from None


class Market:
    def __init__(self, artifact_dir):
        rows = json.loads((Path(artifact_dir) / "sessions.json").read_text(encoding="utf-8"))
        self.sessions = sorted(rows, key=lambda row: row["date"])

    def query(self, start=None, end=None):
        lo, hi = parse_date(start, "start"), parse_date(end, "end")
        if lo and hi and lo > hi:
            raise ValidationError("'start' must not be after 'end'")
        return [row for row in self.sessions if (lo is None or row["date"] >= lo) and (hi is None or row["date"] <= hi)]
