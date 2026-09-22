"""Small-sample rate statistics for the report tables and plots."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Rate:
    """A binomial rate with its 95% Wilson interval."""

    hits: int
    n: int

    @property
    def value(self) -> float:
        """The point estimate ``hits / n`` (NaN when ``n == 0``)."""
        return self.hits / self.n if self.n else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        """The 95% Wilson interval."""
        return wilson_interval(self.hits, self.n)

    def fmt(self) -> str:
        """``0.50 [0.15, 0.85] (n=4)``, or ``—`` when empty."""
        if not self.n:
            return "—"
        lo, hi = self.interval
        return f"{self.value:.2f} [{lo:.2f}, {hi:.2f}] (n={self.n})"


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — well-behaved at 0/n and n/n, unlike the normal approximation."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
