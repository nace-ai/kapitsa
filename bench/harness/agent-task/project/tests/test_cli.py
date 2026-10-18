import io
import os
import tempfile
import unittest

from ledger.cli import main

DATA = os.path.join(os.path.dirname(__file__), "data", "household.csv")


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_balance(self):
        self.assertEqual(run("balance", DATA), (0, "Balance: $9,864.08\n", ""))
        self.assertEqual(run("balance", DATA, "--category", "food"), (0, "Balance: -$434.40\n", ""))
        self.assertEqual(run("balance", DATA, "--tag", "fixed")[1], "Balance: -$5,905.00\n")

    def test_monthly(self):
        code, out, _ = run("monthly", DATA)
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines(), ["Month          Net",
                                            "-------  ---------",
                                            "2025-12  $2,209.70",
                                            "2026-01  $2,515.75",
                                            "2026-02  $2,238.70",
                                            "2026-03  $2,899.93"])

    def test_monthly_date_range(self):
        code, out, _ = run("monthly", DATA, "--since", "2026-01-15", "--until", "2026-02-28")
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[2:], ["2026-01   -$220.05", "2026-02  $2,238.70"])

    def test_monthly_bad_date(self):
        with self.assertRaises(SystemExit) as cm:
            run("monthly", DATA, "--since", "January")
        self.assertEqual(cm.exception.code, 2)

    def test_top(self):
        self.assertEqual(run("top", DATA, "-n", "2")[1], "1. housing: $5,800.00\n2. food: $434.40\n")

    def test_errors(self):
        code, out, err = run("balance", os.path.join(tempfile.gettempdir(), "no-such-ledger.csv"))
        self.assertEqual((code, out), (1, ""))
        self.assertTrue(err.startswith("ledger: "))
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
            fh.write("date,description,amount\n2026-01-01,Broken,twelve\n")
        try:
            code, _, err = run("balance", fh.name)
            self.assertEqual(code, 1)
            self.assertIn("line 2", err)
        finally:
            os.unlink(fh.name)


if __name__ == "__main__":
    unittest.main()
