"""Command line: `python3 -m ledger <command> <file> [options]`.

    balance FILE [--category C] [--tag T]      net balance of the matching entries
    monthly FILE [--since DATE] [--until DATE] net amount per month, as a table
    top FILE [-n N]                            the categories with the most spending

Dates are YYYY-MM-DD. Exit status 0 on success, 1 when the file cannot be read or parsed.
"""

from __future__ import annotations

import argparse
import sys

from . import parser as ledger_parser
from . import report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ledger", description="Summaries of a CSV ledger.")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("balance", help="net balance")
    b.add_argument("file")
    b.add_argument("--category")
    b.add_argument("--tag")

    m = sub.add_parser("monthly", help="net amount per month")
    m.add_argument("file")

    t = sub.add_parser("top", help="categories with the most spending")
    t.add_argument("file")
    t.add_argument("-n", type=int, default=3)
    return ap


def cmd_balance(args, entries, out) -> int:
    chosen = report.filter_entries(entries, category=args.category, tag=args.tag)
    print(f"Balance: {report.balance(chosen)}", file=out)
    return 0


def cmd_monthly(args, entries, out) -> int:
    rows = [(month, amount) for month, amount in report.monthly_totals(entries).items()]
    if not rows:
        print("No entries.", file=out)
        return 0
    print(report.format_table(rows), file=out)
    return 0


def cmd_top(args, entries, out) -> int:
    spending = report.spending_by_category(entries)
    for i, category in enumerate(list(spending)[:args.n], start=1):
        print(f"{i}. {category}: {spending[category]}", file=out)
    return 0


COMMANDS = {"balance": cmd_balance, "monthly": cmd_monthly, "top": cmd_top}


def main(argv=None, out=None, err=None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        entries = ledger_parser.load(args.file)
    except (OSError, ledger_parser.ParseError) as e:
        print(f"ledger: {e}", file=err)
        return 1
    return COMMANDS[args.command](args, entries, out)
