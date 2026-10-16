"""B3 - Query intent classifier: conceptual vs quantitative.

Rule-based on purpose: the labelled set has only 6 quantitative queries, too
few to train and test a model honestly. "Quantitative" means the question asks
for a count, an amount, a measured value or an extreme (highest/lowest), which
is answered by numbers in the data rather than by explanations. Everything
else is "conceptual" (what, why, how to check, which document).

The existing numeric router (``serving.router.classify_query``) handles the
narrower case "superlative + known SCADA metric" for /ask; this classifier is
broader and domain-independent.
"""

from __future__ import annotations

import re

QUANTITATIVE_PATTERNS: tuple[str, ...] = (
    r"\bhow (many|much|often|long)\b",
    r"\b(number|count|total|sum|amount|percentage|share|proportion|rate) of\b",
    r"\b(average|mean|median|maximum|minimum|max|min)\b",
    r"\b(highest|lowest|largest|smallest|peak|most|least)\b",
    r"\bwhat (was|is|were|are) the (value|level|reading|temperature|power|speed|duration)\b",
    r"\b(kw|kwh|mw|m/s|degc|percent|%)\b",
)
_COMPILED = tuple(re.compile(p) for p in QUANTITATIVE_PATTERNS)


def classify_intent(query: str) -> str:
    """Return "quantitative" or "conceptual"."""
    text = (query or "").lower()
    return "quantitative" if any(p.search(text) for p in _COMPILED) else "conceptual"
