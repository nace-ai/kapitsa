"""Reading ledger files: one transaction per line, comma-separated, with a header line.

    date,description,amount,category,tags
    2026-01-03,Coffee beans,-14.20,food,
    2026-01-04,"Rent, January",-1450.00,housing,monthly fixed
    # lines starting with '#' are comments; blank lines are skipped

Descriptions may be quoted (and then contain commas or doubled quotes). Tags are separated by spaces.
The category defaults to "uncategorized" when the column is empty.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .money import Money

COLUMNS = ("date", "description", "amount", "category", "tags")


@dataclass(frozen=True)
class Entry:
    date: dt.date
    description: str
    amount: Money
    category: str = "uncategorized"
    tags: tuple = field(default_factory=tuple)

    @property
    def is_expense(self) -> bool:
        return self.amount.is_negative()


class ParseError(ValueError):
    """A line that is not a valid transaction; carries the 1-based line number."""

    def __init__(self, lineno: int, message: str):
        super().__init__(f"line {lineno}: {message}")
        self.lineno = lineno


def parse_date(text: str) -> dt.date:
    try:
        return dt.datetime.strptime(text.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"bad date {text!r} (expected YYYY-MM-DD)") from None


def split_fields(line: str) -> list:
    """The fields of one CSV line."""
    return [f.strip() for f in line.rstrip("\n").split(",")]


def parse_line(line: str, lineno: int = 1, currency: str = "USD") -> Entry:
    fields = split_fields(line)
    if len(fields) < 3:
        raise ParseError(lineno, f"expected at least 3 fields, got {len(fields)}")
    if len(fields) > len(COLUMNS):
        raise ParseError(lineno, f"expected at most {len(COLUMNS)} fields, got {len(fields)}")
    fields += [""] * (len(COLUMNS) - len(fields))
    date_s, description, amount_s, category, tags_s = fields
    try:
        date = parse_date(date_s)
        amount = Money.parse(amount_s, currency)
    except ValueError as e:
        raise ParseError(lineno, str(e)) from None
    return Entry(date=date, description=description, amount=amount,
                 category=category.lower() or "uncategorized", tags=tuple(tags_s.split()))


def parse_text(text: str, currency: str = "USD") -> list:
    """All entries of a ledger file's text, in file order."""
    entries = []
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("date,"):
        lines = lines[1:]
        start = 2
    else:
        start = 1
    for lineno, line in enumerate(lines, start=start):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        entries.append(parse_line(line, lineno, currency))
    return entries


def load(path: str, currency: str = "USD") -> list:
    with open(path, encoding="utf-8") as fh:
        return parse_text(fh.read(), currency)
