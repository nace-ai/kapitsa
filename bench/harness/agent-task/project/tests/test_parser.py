import datetime as dt
import os
import unittest

from ledger.money import Money
from ledger.parser import Entry, ParseError, load, parse_line, parse_text

DATA = os.path.join(os.path.dirname(__file__), "data", "household.csv")


class ParseLineTest(unittest.TestCase):
    def test_simple_line(self):
        e = parse_line("2026-01-03,Coffee beans,-14.20,Food,")
        self.assertEqual(e, Entry(dt.date(2026, 1, 3), "Coffee beans", Money(-1420), "food", ()))
        self.assertTrue(e.is_expense)

    def test_optional_columns(self):
        e = parse_line("2026-01-05,Salary,3200")
        self.assertEqual(e.category, "uncategorized")
        self.assertEqual(e.tags, ())
        self.assertFalse(e.is_expense)

    def test_tags(self):
        e = parse_line("2026-01-04,Gym,-35,health,monthly fixed")
        self.assertEqual(e.tags, ("monthly", "fixed"))

    def test_quoted_description_with_comma(self):
        e = parse_line('2026-01-04,"Rent, January",-1450.00,housing,monthly')
        self.assertEqual(e.description, "Rent, January")
        self.assertEqual(e.amount, Money(-145000))
        self.assertEqual(e.category, "housing")
        self.assertEqual(e.tags, ("monthly",))

    def test_quoted_description_with_quotes(self):
        e = parse_line('2026-02-14,"Dinner at ""Luigi\'s"", tip included",-86.40,food,')
        self.assertEqual(e.description, 'Dinner at "Luigi\'s", tip included')
        self.assertEqual(e.amount, Money(-8640))

    def test_accounting_amount(self):
        e = parse_line("2026-02-01,Insurance,(120.00),insurance,")
        self.assertEqual(e.amount, Money(-12000))

    def test_errors_carry_line_numbers(self):
        with self.assertRaises(ParseError) as cm:
            parse_line("2026-13-01,Bad month,-1,misc,", lineno=7)
        self.assertEqual(cm.exception.lineno, 7)
        with self.assertRaises(ParseError):
            parse_line("2026-01-01,Only two fields")
        with self.assertRaises(ParseError):
            parse_line("2026-01-01,Too,1,many,fields,here")


class ParseTextTest(unittest.TestCase):
    def test_header_comments_and_blank_lines(self):
        text = "date,description,amount,category,tags\n# opening\n\n2026-01-01,Gift,50,income,\n2026-01-02,Bus,-2.75,transport,\n"
        entries = parse_text(text)
        self.assertEqual([e.description for e in entries], ["Gift", "Bus"])

    def test_line_numbers_count_the_header(self):
        text = "date,description,amount,category,tags\n2026-01-01,Gift,50,income,\n2026-01-02,Bus,lots,transport,\n"
        with self.assertRaises(ParseError) as cm:
            parse_text(text)
        self.assertEqual(cm.exception.lineno, 3)

    def test_sample_file(self):
        entries = load(DATA)
        self.assertEqual(len(entries), 24)
        self.assertEqual(entries[0].date, dt.date(2025, 12, 1))
        self.assertEqual(entries[-1].date, dt.date(2026, 3, 28))
        rent = [e for e in entries if e.category == "housing"]
        self.assertEqual(len(rent), 4)
        self.assertTrue(all(e.amount == Money(-145000) for e in rent))


if __name__ == "__main__":
    unittest.main()
