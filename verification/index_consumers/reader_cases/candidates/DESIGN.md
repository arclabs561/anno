# Cross-spelling candidate retrieval

Frozen before running candidate policies. The existing native AjMC results were already public in this repository. This is a follow-up on the same official test material. No rule or operating point is selected from the volume results.

## Eligibility and rules

All gold outer literal PER mentions are queryable, including unlabeled and NIL occurrences in the gallery. Score known-QID queries. The native resolver receives unbound supplied singleton PER tracks at its existing default. The baseline returns the query track’s native group in the allowed gallery. Each alternative keeps those hits and may suggest additional source occurrences. Use only gallery spellings and opaque native group IDs. Preserve original spellings and separate suggested occurrences so a caller can discard them.

1. **Fold.** Apply NFKC, casefold, collapse whitespace, harmonize curly apostrophes and typographic hyphens, and remove diacritics on Latin letters. Ignore a single terminal period on an otherwise alphabetic word of at least three letters.
2. **Expanded.** Try the fold tier first. If it has no extra group, try a single word explicitly ending in a period whose folded prefix of at least three letters matches the start of a strictly longer unabbreviated word. Try both directions. If that tier has no group, allow a single insertion, deletion or substitution between single alphabetic words of at least five letters in the same script.
3. Ignore the query’s native group when proposing extras. At the first tier with matching donor occurrences, accept only if every matching occurrence belongs to one opaque native group. Multiple groups trigger abstention at that tier. Do not use frequency to break ties or expand through accepted suggestions. Return only directly matching occurrences.

The full-page comparison uses all pages in one native run and excludes the query page from its gallery. For three held-out volume folds, each query page runs independently against the other two volumes. Other pages from the query volume cannot create native links or support its candidate spellings. There is no learned calibration or identifiable author-independent language effect.

## Measures

Report before/after counts for shown queries, eligible queries with any correct hit, eligible queries with their exact supported occurrence set and no extra unscored hit, queries with a verifiably wrong hit and queries with an unscored hit. Count wrong returned source pages separately from wrong occurrences on otherwise supported pages. For added occurrences report correct, wrong and unscored, as well as how many correct and wrong additions differ in raw spelling. Report the first deterministic examples of benefits and harms in anchor order after scoring. A page containing only unscored mentions remains unevaluable for page-level error. Print each held-out volume and pooled folds. Keep the three volumes out of any claim of statistical independence for individual mentions.
