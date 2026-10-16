"""C3 - Rule-based prompt-injection detector for user queries and retrieved text.

Each pattern describes a *kind* of instruction that has no place in a maintenance question or a knowledge
document (overriding instructions, forcing a fixed output, claiming system authority, changing the model's
role, hiding sources, extracting the prompt). A text is flagged when the summed pattern weights reach
``THRESHOLD``: one strong signal (weight 2) or two weak ones. The patterns and threshold were fixed before
the evaluation in experiments/detector_evaluation.py and were not tuned on its results.

Known blind spots (by construction of a keyword detector): other languages, spaced or encoded text, and
polite requests without instruction vocabulary ("pretend you are ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

THRESHOLD = 2


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: str
    weight: int


PATTERNS = (
    Pattern("override_instructions",
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,60}\b(instructions?|prompts?|rules|context|"
            r"evidence|guidelines|documents?)\b", 2),
    Pattern("override_evidence", r"\bregardless of (the |any )?(evidence|context|instructions|data|documents?)\b", 2),
    Pattern("prompt_exfiltration",
            r"\b(reveal|print|output|show|repeat|translate)\b[^.\n]{0,30}\b(your|the|hidden|system)\s+"
            r"(instructions|prompt|rules)\b", 2),
    Pattern("forced_output",
            r"\b(reply|respond|answer|output|say|print|return)\s+(only|exactly|just)\b|\b(just|only|always)\s+"
            r"(say|reply|respond|answer|output)\b", 1),
    Pattern("fake_authority", r"\b(system|admin|administrator|developer)\s*(note|prompt|message|override|instruction)s?\b",
            1),
    Pattern("role_change", r"\byou are now\b|\bact as\b|\bpretend (you are|to be)\b|\bfrom now on\b|"
            r"\bnew (role|instructions?|task|system prompt)\b", 1),
    Pattern("concealment", r"\b(do not|don't|never)\s+(cite|mention|reveal|tell|disclose)\b", 1),
    Pattern("role_markup", r"</?\s*(system|assistant|user)\s*>|(^|\n)\s*(system|assistant)\s*:", 1),
)
_COMPILED = [(p, re.compile(p.regex, re.IGNORECASE)) for p in PATTERNS]


def _normalise(text: str) -> str:
    text = re.sub(r"[​-‏⁠﻿]", "", text or "")       # zero-width characters
    return re.sub(r"[ \t]+", " ", text)


def injection_signals(text: str) -> list[str]:
    t = _normalise(text)
    return [p.name for p, rx in _COMPILED if rx.search(t)]


def injection_score(text: str) -> int:
    weights = {p.name: p.weight for p in PATTERNS}
    return sum(weights[n] for n in injection_signals(text))


def is_injection(text: str, threshold: int = THRESHOLD) -> bool:
    return injection_score(text) >= threshold
