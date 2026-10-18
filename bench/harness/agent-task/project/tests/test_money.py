import unittest

from ledger.money import Money, total


class ParseTest(unittest.TestCase):
    def test_plain_amounts(self):
        self.assertEqual(Money.parse("12.50"), Money(1250))
        self.assertEqual(Money.parse("-3"), Money(-300))
        self.assertEqual(Money.parse("+7"), Money(700))
        self.assertEqual(Money.parse(" 4.5 "), Money(450))

    def test_symbols_and_thousands(self):
        self.assertEqual(Money.parse("$7.05"), Money(705))
        self.assertEqual(Money.parse("1,234.56"), Money(123456))
        self.assertEqual(Money.parse("-$1,000"), Money(-100000))
        self.assertEqual(Money.parse("€3.10", "EUR"), Money(310, "EUR"))

    def test_accounting_negatives(self):
        self.assertEqual(Money.parse("(12.50)"), Money(-1250))
        self.assertEqual(Money.parse("($1,200.00)"), Money(-120000))
        self.assertEqual(Money.parse(" (0.99) "), Money(-99))

    def test_rounds_half_away_from_zero(self):
        self.assertEqual(Money.parse("2.675"), Money(268))
        self.assertEqual(Money.parse("1.005"), Money(101))
        self.assertEqual(Money.parse("0.125"), Money(13))
        self.assertEqual(Money.parse("-0.005"), Money(-1))
        self.assertEqual(Money.parse("19.994"), Money(1999))

    def test_rejects_garbage(self):
        for text in ["", "abc", "12.5.1", "()", "$"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    Money.parse(text)


class ArithmeticTest(unittest.TestCase):
    def test_add_sub_neg(self):
        self.assertEqual(Money(150) + Money(250), Money(400))
        self.assertEqual(Money(150) - Money(250), Money(-100))
        self.assertEqual(-Money(150), Money(-150))
        self.assertEqual(abs(Money(-150)), Money(150))

    def test_currency_mismatch(self):
        with self.assertRaises(ValueError):
            Money(100, "USD") + Money(100, "EUR")

    def test_total(self):
        self.assertEqual(total([Money(1), Money(2), Money(-5)]), Money(-2))
        self.assertEqual(total([]), Money(0))

    def test_split(self):
        self.assertEqual(Money(1000).split(3), [Money(334), Money(333), Money(333)])
        self.assertEqual(Money(-100).split(3), [Money(-34), Money(-33), Money(-33)])
        self.assertEqual(sum(m.cents for m in Money(12345).split(7)), 12345)


class FormatTest(unittest.TestCase):
    def test_format(self):
        self.assertEqual(Money(123450).format(), "$1,234.50")
        self.assertEqual(Money(-305).format(), "-$3.05")
        self.assertEqual(Money(99, "EUR").format(), "€0.99")
        self.assertEqual(str(Money(5, "CHF")), "CHF 0.05")


if __name__ == "__main__":
    unittest.main()
