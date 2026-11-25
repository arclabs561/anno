# SPDX-License-Identifier: MIT OR Apache-2.0
"""Suggest visible source occurrences from a donor gallery using only names and opaque groups."""

from collections import defaultdict
from dataclasses import dataclass
from unicodedata import category, name, normalize


PUNCTUATION = str.maketrans(
    {"’": "'", "‘": "'", "‐": "-", "‑": "-", "‒": "-", "–": "-"}
)


def spelling(surface):
    folded = normalize("NFKC", surface).casefold().translate(PUNCTUATION)
    chars = []
    latin = False
    for char in normalize("NFD", folded):
        if category(char).startswith("M"):
            if not latin:
                chars.append(char)
        else:
            latin = name(char, "").startswith("LATIN ")
            chars.append(char)
    word = " ".join(normalize("NFC", "".join(chars)).split())
    abbreviated = word.endswith(".") and len(word) >= 4 and word[:-1].isalpha()
    return (word[:-1] if abbreviated else word, abbreviated)


def single_script(word):
    scripts = {name(ch, "").split(" ", 1)[0] for ch in word}
    return scripts.pop() if len(scripts) == 1 else None


def one_edit(a, b):
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = sorted((a, b), key=len)
    if len(long) != len(short) + 1:
        return False
    i = next((i for i, char in enumerate(short) if char != long[i]), len(short))
    return short[i:] == long[i + 1 :]


def abbreviation(a, b):
    if a[1] == b[1]:
        return False
    (prefix, _), (whole, _) = (a, b) if a[1] else (b, a)
    return whole.isalpha() and len(prefix) < len(whole) and whole.startswith(prefix)


@dataclass(frozen=True)
class Suggestion:
    tier: str
    decision: str
    hits: frozenset


class CandidateIndex:
    """Map stable occurrence keys to (raw spelling, opaque native group) in a permitted gallery."""

    def __init__(self, records):
        self.forms = defaultdict(list)
        for occurrence, (raw, group) in records.items():
            self.forms[spelling(raw)].append((occurrence, group))

    def suggest(self, query_surface, native_group, variant):
        if variant == "native":
            return Suggestion("native", "none", frozenset())
        if variant not in {"fold", "expanded"}:
            raise ValueError("unknown candidate variant")
        query = spelling(query_surface)
        tiers = ["fold"] if variant == "fold" else ["fold", "abbreviation", "edit"]
        for tier in tiers:
            records = []
            for form, occurrences in self.forms.items():
                if tier == "fold":
                    matches = form[0] == query[0]
                elif tier == "abbreviation":
                    matches = abbreviation(query, form)
                else:
                    a, b = query[0], form[0]
                    matches = (
                        min(len(a), len(b)) >= 5
                        and a.isalpha()
                        and b.isalpha()
                        and single_script(a) is not None
                        and single_script(a) == single_script(b)
                        and one_edit(a, b)
                    )
                if matches:
                    records.extend(
                        (key, group)
                        for key, group in occurrences
                        if group != native_group
                    )
            if records:
                if len({group for _, group in records}) > 1:
                    return Suggestion(tier, "ambiguous", frozenset())
                return Suggestion(
                    tier, "suggested", frozenset(key for key, _ in records)
                )
        return Suggestion("none", "none", frozenset())
