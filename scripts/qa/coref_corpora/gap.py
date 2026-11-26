"""English GAP article-disjoint pronoun controls, using the pinned publisher test TSV.

Source: https://github.com/google-research-datasets/gap-coreference/tree/83135f26025cdd705c04f77e940d0c9201116b80
The ordinary panel samples articles without reading reference decisions. The challenge
samples other articles with a reference of neither offered name. Score these separately.
"""

import csv
import hashlib
import io
import unicodedata
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

SOURCE_SHA256 = "1c35e36d5b14f6313ec3f6cd67b275de282595dd59e59390e00cfff9897a6819"
SEED = "anno-gap-review-reserve-20261126-v1"
COLUMNS = [
    "ID", "Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref",
    "B", "B-offset", "B-coref", "URL",
]


def _rank(salt, value):
    return hashlib.sha256(f"{salt}/{value}".encode()).hexdigest()


def _article(row):
    url = urlsplit(row["URL"])
    if not url.hostname or not url.path:
        raise ValueError("Missing GAP article identity")
    return url.hostname + unicodedata.normalize("NFC", unquote(url.path)).replace(" ", "_")


def _historical_rows(rows):
    # The published panel and the independent decision audit share a fixed order
    # (scripts/qa/audit_gap_decisions.py). Its holdout never reuses a published URL.
    ordered = sorted(rows, key=lambda r: _rank("anno-gap-v1", r["ID"]))
    published = ordered[:128]
    urls = {r["URL"] for r in published}
    decision_holdout = []
    for row in ordered[128:]:
        if row["URL"] not in urls:
            decision_holdout.append(row)
            urls.add(row["URL"])
            if len(decision_holdout) == 128:
                break

    # The replication and proposal panels were frozen independently of that
    # decision audit. Recreate their original exclusions before taking the union.
    other_panels = list(published)
    urls = {r["URL"] for r in published}
    texts = {r["Text"] for r in published}
    for salt in ("anno-identity-article-replication-2026-11-25", "anno-proposal-recovery-v1"):
        selected = []
        for row in sorted(rows, key=lambda r: _rank(salt, r["ID"])):
            if row["URL"] in urls or row["Text"] in texts:
                continue
            urls.add(row["URL"])
            texts.add(row["Text"])
            selected.append(row)
            if len(selected) == 128:
                break
        other_panels.extend(selected)
    return other_panels + decision_holdout


def _draw(rows, cohort, count):
    by_article = {}
    for row in rows:
        by_article.setdefault(_article(row), []).append(row)
    salt = f"{SEED}/{cohort}"
    articles = sorted(by_article, key=lambda a: _rank(salt + "/article", a))
    return [
        min(by_article[a], key=lambda r: _rank(salt + "/row", r["ID"]))
        for a in articles[:count]
    ]


def _span(row, field):
    start = int(row[field + "-offset"])
    surface = row[field]
    end = start + len(surface)
    if start < 0 or not surface or row["Text"][start:end] != surface:
        raise ValueError(f"{row['ID']}: incorrect {field} span")
    return [start, end]


def _choice(row, a, b):
    labels = row[a + "-coref"], row[b + "-coref"]
    answers = {
        ("TRUE", "FALSE"): "A", ("FALSE", "TRUE"): "B",
        ("FALSE", "FALSE"): "neither", ("TRUE", "TRUE"): "both",
    }
    if labels not in answers:
        raise ValueError(f"{row['ID']}: invalid GAP coreference label")
    return answers[labels]


def _case(row, cohort):
    if int(_rank(SEED + "/display", row["ID"]), 16) % 2:
        a, b = "B", "A"
    else:
        a, b = "A", "B"
    article_url = row["URL"]
    page = urlsplit(article_url)
    if page.hostname != "en.wikipedia.org" or not page.path.startswith("/wiki/"):
        raise ValueError("Expected an English Wikipedia article URL")
    title = quote(unquote(page.path.removeprefix("/wiki/")), safe="")
    return {
        "id": _rank(SEED + "/case", row["ID"])[:16],
        "group": _rank("gap-article", _article(row))[:20],
        "text": row["Text"],
        "kind": "choice",
        "spans": {"target": _span(row, "Pronoun"), "A": _span(row, a), "B": _span(row, b)},
        "gold": _choice(row, a, b),
        "tags": [cohort],
        "provenance": {
            "source_row_id": row["ID"],
            "original_article_url": article_url,
            "article_history_url": f"https://en.wikipedia.org/w/index.php?title={title}&action=history",
        },
    }


def cases(source: Path):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected publisher gap-test.tsv at the pinned SHA-256")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), delimiter="\t")
    if reader.fieldnames != COLUMNS:
        raise ValueError("Unexpected GAP column names")
    rows = list(reader)
    if len(rows) != 2000 or len({r["ID"] for r in rows}) != len(rows):
        raise ValueError("Expected 2,000 distinct GAP test rows")
    prior = _historical_rows(rows)
    used = {_article(r) for r in prior}
    texts = {r["Text"] for r in prior}
    remaining = [r for r in rows if _article(r) not in used and r["Text"] not in texts]
    ordinary = _draw(remaining, "ordinary", 64)
    used.update(_article(r) for r in ordinary)
    texts.update(r["Text"] for r in ordinary)
    outside = [r for r in remaining if _article(r) not in used and r["Text"] not in texts
               and _choice(r, "A", "B") == "neither"]
    challenge = _draw(outside, "challenge", 24)
    tagged = [(r, "ordinary") for r in ordinary] + [(r, "challenge") for r in challenge]
    if (len(ordinary), len(challenge), len({_article(r) for r, _ in tagged}),
            len({r["Text"] for r, _ in tagged})) != (64, 24, 88, 88):
        raise ValueError("Expected 88 text- and article-disjoint GAP cases")
    for row, cohort in sorted(tagged, key=lambda rc: _rank(SEED + "/order", rc[0]["ID"])):
        yield _case(row, cohort)
