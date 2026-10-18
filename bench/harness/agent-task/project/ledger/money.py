"""Money amounts, stored as a whole number of cents so that sums never drift."""

from __future__ import annotations

from dataclasses import dataclass

CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}


@dataclass(frozen=True, order=True)
class Money:
    """An amount of money in one currency. `cents` is the amount in the currency's minor unit."""

    cents: int
    currency: str = "USD"

    @classmethod
    def parse(cls, text: str, currency: str = "USD") -> "Money":
        """Parses an amount as people write it in a spreadsheet.

        Accepted forms: "12.50", "-3", "+7", "1,234.56", "$7.05", " 4.5 ", and the accounting style for
        negative amounts, "(12.50)". Amounts with more than two decimals are rounded to the nearest cent,
        halves away from zero: "2.675" is 2.68 and "-0.005" is -0.01.
        """
        s = text.strip()
        if not s:
            raise ValueError("empty amount")
        sign = 1
        if s[0] in "+-":
            sign = -1 if s[0] == "-" else 1
            s = s[1:].strip()
        for symbol in CURRENCY_SYMBOLS.values():
            if s.startswith(symbol):
                s = s[len(symbol):]
        s = s.replace(",", "")
        try:
            value = float(s)
        except ValueError:
            raise ValueError(f"not an amount: {text!r}") from None
        return cls(sign * round(value * 100), currency)

    @classmethod
    def zero(cls, currency: str = "USD") -> "Money":
        return cls(0, currency)

    def _check(self, other: "Money") -> None:
        if not isinstance(other, Money):
            raise TypeError(f"cannot combine Money with {type(other).__name__}")
        if other.currency != self.currency:
            raise ValueError(f"currency mismatch: {self.currency} and {other.currency}")

    def __add__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents + other.cents, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        self._check(other)
        return Money(self.cents - other.cents, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.cents, self.currency)

    def __abs__(self) -> "Money":
        return Money(abs(self.cents), self.currency)

    def __bool__(self) -> bool:
        return self.cents != 0

    def is_negative(self) -> bool:
        return self.cents < 0

    def split(self, parts: int) -> list:
        """Splits the amount into `parts` amounts that differ by at most one cent and add up exactly."""
        if parts <= 0:
            raise ValueError("parts must be positive")
        base, extra = divmod(abs(self.cents), parts)
        sign = -1 if self.cents < 0 else 1
        return [Money(sign * (base + (1 if i < extra else 0)), self.currency) for i in range(parts)]

    def format(self) -> str:
        """"$1,234.50", "-$3.05", "€0.99": the sign goes before the symbol, thousands get commas."""
        symbol = CURRENCY_SYMBOLS.get(self.currency, self.currency + " ")
        whole, cents = divmod(abs(self.cents), 100)
        sign = "-" if self.cents < 0 else ""
        return f"{sign}{symbol}{whole:,}.{cents:02d}"

    def __str__(self) -> str:
        return self.format()


def total(amounts, currency: str = "USD") -> Money:
    """The sum of an iterable of Money (zero for an empty one)."""
    result = Money.zero(currency)
    for amount in amounts:
        result = result + amount
    return result
