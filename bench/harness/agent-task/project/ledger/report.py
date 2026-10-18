"""Summaries of ledger entries: balances, monthly totals, spending by category."""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict

from .money import Money, total


def filter_entries(entries, since=None, until=None, category=None, tag=None):
    """Entries on or after `since` and on or before `until` (dates), in `category`, with `tag`; None = any."""
    out = []
    for e in entries:
        if since is not None and e.date < since:
            continue
        if until is not None and e.date > until:
            continue
        if category is not None and e.category != category.lower():
            continue
        if tag is not None and tag not in e.tags:
            continue
        out.append(e)
    return out


def balance(entries, currency: str = "USD") -> Money:
    """Income minus expenses."""
    return total((e.amount for e in entries), currency)


def month_key(date: dt.date) -> str:
    return f"{date.year:04d}-{date.month:02d}"


def monthly_totals(entries, currency: str = "USD") -> "OrderedDict[str, Money]":
    """Net amount per calendar month, keyed "YYYY-MM", in chronological order. Months without entries
    are left out."""
    sums = {}
    for e in entries:
        key = month_key(e.date)
        sums[key] = sums.get(key, Money.zero(currency)) + e.amount
    ordered = sorted(sums, key=lambda k: int(k.split("-")[1]))
    return OrderedDict((k, sums[k]) for k in ordered)


def spending_by_category(entries, currency: str = "USD") -> "OrderedDict[str, Money]":
    """Total spent per category (expenses only, as positive amounts), largest first; ties by name."""
    sums = {}
    for e in entries:
        if e.is_expense:
            sums[e.category] = sums.get(e.category, Money.zero(currency)) + abs(e.amount)
    ordered = sorted(sums.items(), key=lambda kv: (-kv[1].cents, kv[0]))
    return OrderedDict(ordered)


def top_categories(entries, n: int = 3, currency: str = "USD") -> list:
    """The `n` categories with the most spending, largest first."""
    return list(spending_by_category(entries, currency))[:n]


def running_balance(entries, opening: Money = None, currency: str = "USD") -> list:
    """(date, balance after that day) for every day with entries, in date order."""
    current = opening if opening is not None else Money.zero(currency)
    by_day = OrderedDict()
    for e in sorted(entries, key=lambda e: e.date):
        current = current + e.amount
        by_day[e.date] = current
    return list(by_day.items())


def format_table(rows, headers=("Month", "Net")) -> str:
    """A plain-text table with a header row, the first column left-aligned and the rest right-aligned."""
    cells = [tuple(str(c) for c in headers)] + [tuple(str(c) for c in r) for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = []
    for j, r in enumerate(cells):
        parts = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(r))]
        lines.append("  ".join(parts).rstrip())
        if j == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
