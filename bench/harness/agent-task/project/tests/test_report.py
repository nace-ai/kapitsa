import datetime as dt
import os
import unittest

from ledger import report
from ledger.money import Money
from ledger.parser import Entry, load

DATA = os.path.join(os.path.dirname(__file__), "data", "household.csv")


def entry(day, amount, category="misc", tags=()):
    return Entry(dt.date.fromisoformat(day), "x", Money(amount), category, tuple(tags))


class FilterTest(unittest.TestCase):
    def setUp(self):
        self.entries = [entry("2026-01-31", -100, "food"), entry("2026-02-01", -200, "rent", ["fixed"]),
                        entry("2026-02-28", 300, "income"), entry("2026-03-01", -50, "food")]

    def test_dates_are_inclusive(self):
        got = report.filter_entries(self.entries, since=dt.date(2026, 2, 1), until=dt.date(2026, 2, 28))
        self.assertEqual([e.amount.cents for e in got], [-200, 300])

    def test_category_and_tag(self):
        self.assertEqual(len(report.filter_entries(self.entries, category="FOOD")), 2)
        self.assertEqual(len(report.filter_entries(self.entries, tag="fixed")), 1)


class MonthlyTest(unittest.TestCase):
    def test_chronological_across_years(self):
        entries = [entry("2026-01-05", 1000), entry("2025-11-30", -250), entry("2025-12-31", 400),
                   entry("2026-01-20", -100), entry("2024-12-01", 1)]
        totals = report.monthly_totals(entries)
        self.assertEqual(list(totals), ["2024-12", "2025-11", "2025-12", "2026-01"])
        self.assertEqual(totals["2026-01"], Money(900))

    def test_sample_file(self):
        totals = report.monthly_totals(load(DATA))
        self.assertEqual(list(totals.items()), [("2025-12", Money(220970)), ("2026-01", Money(251575)),
                                                ("2026-02", Money(223870)), ("2026-03", Money(289993))])


class SpendingTest(unittest.TestCase):
    def test_by_category(self):
        spending = report.spending_by_category(load(DATA))
        self.assertEqual(list(spending.items())[:4], [("housing", Money(580000)), ("food", Money(43440)),
                                                      ("insurance", Money(31200)), ("utilities", Money(29870))])
        self.assertNotIn("income", spending)

    def test_top_categories(self):
        self.assertEqual(report.top_categories(load(DATA), 3), ["housing", "food", "insurance"])

    def test_balance(self):
        self.assertEqual(report.balance(load(DATA)), Money(986408))


class RunningBalanceTest(unittest.TestCase):
    def test_one_row_per_day(self):
        entries = [entry("2026-01-02", -100), entry("2026-01-01", 1000), entry("2026-01-02", -50)]
        rows = report.running_balance(entries, opening=Money(500))
        self.assertEqual(rows, [(dt.date(2026, 1, 1), Money(1500)), (dt.date(2026, 1, 2), Money(1350))])


class TableTest(unittest.TestCase):
    def test_format_table(self):
        text = report.format_table([("2026-01", Money(123456)), ("2026-02", Money(-5))])
        self.assertEqual(text.splitlines(), ["Month          Net",
                                             "-------  ---------",
                                             "2026-01  $1,234.56",
                                             "2026-02     -$0.05"])


if __name__ == "__main__":
    unittest.main()
