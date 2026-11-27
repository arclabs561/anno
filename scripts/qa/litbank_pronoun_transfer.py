# SPDX-License-Identifier: MIT OR Apache-2.0
"""Prepare a private, independently labeled literary transfer panel in GAP TSV form."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from coref_corpora import litbank
from native_gap_neutral_type import ROOT, read_sources


# Complement of the previous 27-work competing-name and ten-work fold-0 panels.
WORKS = {
    24, 32, 44, 45, 60, 74, 76, 95, 110, 113, 120, 145, 155, 158, 160, 171, 174, 208, 209,
    215, 217, 219, 345, 351, 432, 434, 472, 514, 543, 550, 599, 766, 768, 805, 969, 1206,
    1245, 1260, 1342, 1400, 1695, 2005, 2084, 2166, 2641, 2775, 2814, 2852, 2891, 3457,
    4051, 4217, 4276, 4300, 5230, 5348, 6053, 9830, 11231, 12677, 15265, 18581, 41286,
}
SOURCE_SHA256 = "2b53a899c9c5f0d8a4b59ce75a5c7ee67f9a8f8fb424701aebe8a88b2d1e3706"
PRONOUNS = {"he", "she", "him", "her", "his", "hers"}
FIELDS = ("ID", "Text", "Pronoun", "Pronoun-offset", "A", "A-offset", "A-coref",
          "B", "B-offset", "B-coref", "URL")


def choices(work, text, mentions, counts):
    names = sorted((m for m in mentions if m.tag == "PROP_PER"),
                   key=lambda m: (m.span[1], m.span[0]), reverse=True)
    for pronoun in sorted(mentions):
        start, end = pronoun.span
        form = text[start:end]
        if pronoun.tag != "PRON_PER" or form.casefold() not in PRONOUNS:
            continue
        counts["targets"] += 1
        left = max(0, start - 500)
        previous = [m for m in names if left <= m.span[0] and m.span[1] <= start]
        if not previous:
            counts["no_prior_name"] += 1
            continue
        nearest = previous[0]
        nearest_form = text[slice(*nearest.span)].casefold()
        other = next((m for m in previous[1:] if m.span[1] <= nearest.span[0]
                      and text[slice(*m.span)].casefold() != nearest_form), None)
        if other is None:
            counts["no_second_distinct_name"] += 1
            continue
        # Read chain identity only after selecting the two options by position and form.
        if nearest.chain == other.chain:
            counts["same_candidate_identity"] += 1
            counts["both_true_excluded"] += nearest.chain == pronoun.chain
            continue
        row = {"ID": f"litbank-{work}-{start}-{end}", "Text": text[left:end + 80],
               "URL": f"https://www.gutenberg.org/ebooks/{work}",
               "Pronoun": form, "Pronoun-offset": start - left}
        for key, mention in (("A", other), ("B", nearest)):
            a, b = mention.span
            row[key] = text[a:b]
            row[f"{key}-offset"] = a - left
            row[f"{key}-coref"] = "TRUE" if mention.chain == pronoun.chain else "FALSE"
        for key in ("A", "B", "Pronoun"):
            a = row[f"{key}-offset"]
            if row["Text"][a:a + len(row[key])] != row[key]:
                raise ValueError("selected mention does not recover its source")
        counts["eligible"] += 1
        yield row


def phrases(text):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + 20]) for i in range(len(words) - 19)}


def prepare(source, gap):
    fold_zero = json.loads((ROOT / "docs/eval/coref-panels.json").read_text())["litbank"]["documents"]
    previous_works = litbank.WORKS | {int(row["id"].split("_")[0]) for row in fold_zero}
    if WORKS & previous_works:
        raise ValueError("transfer works overlap a prior LitBank panel")
    splits, gap_hashes = read_sources(gap)
    gap_urls, gap_phrases, gap_texts = set(), set(), set()
    for split in splits.values():
        for row in split:
            gap_urls.add(row["URL"])
            gap_phrases.update(phrases(row["Text"]))
            gap_texts.add(" ".join(row["Text"].casefold().split()))
    works = litbank.load_works(source, WORKS, SOURCE_SHA256)
    rows, counts, gold = [], Counter(overlap_with_gap=0), Counter()
    for work in sorted(works, key=int):
        text, coref, entities, _ = works[work]
        mentions, _ = litbank._mentions(text, coref, entities)
        for row in choices(work, text, mentions, counts):
            normalized = " ".join(row["Text"].casefold().split())
            if row["URL"] in gap_urls or normalized in gap_texts or phrases(row["Text"]) & gap_phrases:
                counts["overlap_with_gap"] += 1
                continue
            gold["A" if row["A-coref"] == "TRUE" else "B" if row["B-coref"] == "TRUE" else "neither"] += 1
            rows.append(row)
    if len({r["ID"] for r in rows}) != len(rows) or not rows:
        raise ValueError("expected distinct nonempty transfer observations")
    if (counts["targets"] != sum(counts[k] for k in ("no_prior_name", "no_second_distinct_name",
                                                      "same_candidate_identity", "eligible"))
            or len(rows) + counts["overlap_with_gap"] != counts["eligible"]):
        raise ValueError("selection denominators differ")
    return rows, {"litbank_brat_manifest_sha256": SOURCE_SHA256, "gap_sha256": gap_hashes,
                  "source_works": len(works), "excluded_prior_works": len(previous_works),
                  "selected_works": len({r["URL"] for r in rows}), "selected_rows": len(rows),
                  "selection": dict(counts), "gold": dict(gold)}


def export_panel(destination, rows, selection, expected=None):
    # Stage privately and link only complete, checked bytes to a fresh destination.
    staged = tempfile.NamedTemporaryFile(dir=destination.parent, mode="w", newline="",
                                         encoding="utf-8", delete=False)
    path = Path(staged.name)
    try:
        with staged:
            writer = csv.DictWriter(staged, FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        receipt = dict(selection, panel_tsv_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if expected is not None and receipt != expected:
            raise ValueError("LitBank transfer panel differs from the frozen selection")
        os.link(path, destination)
        return receipt
    finally:
        path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("litbank", type=Path, help="pinned upstream LitBank checkout")
    parser.add_argument("gap", type=Path, help="directory containing all three pinned GAP TSVs")
    parser.add_argument("private_tsv", type=Path, help="new local file with source text and labels")
    parser.add_argument("--check", action="store_true", help="check the frozen panel and aggregate selection")
    args = parser.parse_args()
    expected = json.loads((ROOT / "docs/eval/litbank-pronoun-transfer.json").read_text()) if args.check else None
    rows, selection = prepare(args.litbank, args.gap)
    print(json.dumps(export_panel(args.private_tsv, rows, selection, expected), indent=2, sort_keys=True))
