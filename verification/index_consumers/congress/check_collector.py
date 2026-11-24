# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exercise the pinned external collector's House vote parser using local source files."""
import argparse
from collections import Counter
from contextlib import chdir
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

COLLECTOR = "8184bcb13160da2389dea4c902de67817db681fc"
RUNS = (
    (1, "238b522772f264f3eb59b1ee83f9308f96248a03", "d5600044c18f39e9d263b336b8bc88210773dac127b5529903041dfd8191b812"),
    (1, "c2bd05283e3dbf18dde3134df1710421c959a890", "d5600044c18f39e9d263b336b8bc88210773dac127b5529903041dfd8191b812"),
    (1, "a156ccbd684c23e6974ce84b1438e6101fe9d405", "d5600044c18f39e9d263b336b8bc88210773dac127b5529903041dfd8191b812"),
    (400, "98e8dceb1a11f11efec77b616e6bc344d38e2233", "7c4bf38a7cd038b3aeeed3164f4d3205e09d452740bf8e6594358fcd53fb0408"),
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def study(roll, edition, expected_digest, xml_cache, repo, output_dir):
    from congress.tasks import utils, vote_info
    from lxml import etree

    xml = (xml_cache / f"roll{roll:03}-2023.xml").read_bytes()
    if sha256(xml).hexdigest() != expected_digest:
        raise ValueError(f"wrong Clerk file for roll {roll}")
    output_dir.mkdir()
    books = output_dir / "congress-legislators"
    books.mkdir()
    for book in ("legislators-historical.yaml", "legislators-current.yaml"):
        (books / book).write_bytes(git(repo, "show", f"{edition}:{book}"))
    utils.lookup_legislator_cache = []
    utils.has_congress_legislators_repo = False
    root = etree.fromstring(xml)
    subjects = root.xpath("vote-data/recorded-vote/legislator")
    detail = []
    counts = Counter()
    with chdir(output_dir), patch.object(utils, "lookup_legislator", wraps=utils.lookup_legislator) as lookup:
        def parse():
            vote = {"vote_id": f"h{roll}-118.2023", "congress": 118}
            vote_info.parse_house_vote(root, vote)
            return [v for group in vote["votes"].values() for v in group]
        original = parse()
        expected = [s.attrib["name-id"] for s in subjects]
        assert len(expected) == len(set(expected))
        source_attributions = Counter((s.text, s.attrib["state"], s.attrib["name-id"]) for s in subjects)
        assert source_attributions == Counter((v["display_name"], v["state"], v["id"]) for v in original)
        assert not lookup.called
        # Swap two real nonempty IDs in memory: the ID set and all printed names stay the same.
        pair = {s.text: s for s in subjects if s.text in ("Garcia, Mike", "Garcia, Robert")}
        mike, robert = pair["Garcia, Mike"], pair["Garcia, Robert"]
        mike_id, robert_id = mike.attrib["name-id"], robert.attrib["name-id"]
        mike.set("name-id", robert_id)
        robert.set("name-id", mike_id)
        try:
            swapped = {v["display_name"]: v["id"] for v in parse() if v["display_name"] in pair}
            assert not lookup.called
            assert swapped == {mike.text: robert_id, robert.text: mike_id}
        finally:
            mike.set("name-id", mike_id)
            robert.set("name-id", robert_id)
        for subject in subjects:
            lookup.reset_mock()
            gold = subject.attrib.pop("name-id")
            try:
                voters = parse()
            except Exception as exc:
                if not str(exc).startswith("No bioguide ID for "):
                    raise
                found = None
            else:
                match = [v for v in voters if v["display_name"] == subject.text and v["state"] == subject.attrib["state"]]
                assert len(match) == 1
                found = match[0]["id"]
            finally:
                subject.set("name-id", gold)
            assert lookup.call_count == 1
            counts["missing_or_ambiguous" if found is None else "correct" if gold == found else "wrong"] += 1
            after_opening_update = roll == 1 and edition != RUNS[0][1]
            keep_failure = found != gold and (found is not None or after_opening_update)
            if subject.text in ("Garcia, Robert", "Garcia, Mike", "García (IL)", "Davis (NC)", "Kamlager-Dove", "McEachin") or keep_failure:
                detail.append([subject.text, gold, found])
    return {"roll": roll, "edition": edition, "clean_rows": len(expected), "clean_lookup_calls": 0,
            "withhold_one_at_a_time": dict(counts), "examples": detail,
            "deliberately_swapped_supplied_ids": swapped, "swap_lookup_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("collector", "roster", "clerk_files"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    collector, repo, xml = [getattr(args, k).resolve() for k in ("collector", "roster", "clerk_files")]
    if git(collector, "rev-parse", "HEAD").decode().strip() != COLLECTOR:
        raise ValueError("expected pinned collector revision")
    sys.path.insert(0, str(collector))
    os.environ["UPDATE_CONGRESS_LEGISLATORS"] = "NO"
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory() as work:
        for roll, edition, digest in RUNS:
            result = study(roll, edition, digest, xml, repo, Path(work) / edition)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
