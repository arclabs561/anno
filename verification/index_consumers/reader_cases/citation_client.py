# SPDX-License-Identifier: MIT OR Apache-2.0
"""Follow targets returned by the public reader and verify separately issued staff receipts."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tempfile

from check import HERE, call_json


def inspect_role(ask, case, role, at, policy):
    """Ask for credited targets, then follow each returned target to its visible support."""
    context = {"id": "read", "role": role, "at": at, "policy": policy, "audience": "public"}
    targets = ask({**context, "kind": "credit"})
    receipts = {}
    for target in targets:
        question = {**context, "kind": "cited_credit", "target": target}
        public = ask(question)
        staff = ask({**question, "audience": "staff"})
        staff_by_key = {(row["issuer"], row["id"]): row for row in staff}
        assert public and len(staff_by_key) == len(staff)
        for row in public:
            assert (row["work"], row["role"], row["target"]) == (case["work"], role, target)
            attested = staff_by_key[row["issuer"], row["id"]]
            if row["excerpt"] is not None:
                assert row["excerpt"] == {"quote": attested["excerpt"]["quote"]}
        for row in staff:
            if row["excerpt"] is not None:
                quote = row["excerpt"]["quote"]
                proof = row["excerpt"]["citation"]
                pin, bounds = proof["edition"], proof["range"]
                source = next(s for s in case["sources"] if s["id"] == pin["document"])
                digest = hashlib.sha256(source["text"].encode("utf-8")).hexdigest()
                assert (pin["stream"], pin["source_revision"], pin["source_sha256"]) == (
                    case["id"], source["edition"], digest)
                start, end = bounds["start"], bounds["end"]
                assert 0 <= start < end <= len(source["text"])
                assert source["text"][start:end] == quote
        receipts[target] = public, staff
    return receipts


def check_client(anno, rust):
    suite_path = HERE / "fixtures" / "cases.json"
    cases = {c["id"]: c for c in json.loads(suite_path.read_text())["cases"]}
    plos = cases["plos-role-lists"]
    shared = cases["issuer-local-credit-citations"]
    base = {"id": "read", "role": "engraving", "at": 1, "policy": 1,
            "audience": "public", "kind": "cited_credit", "target": "people:12"}

    def commands(path, case):
        return [[rust, path, "--ask", case["id"]],
                [sys.executable, HERE / "check.py", "--anno", anno,
                 "--cases", path, "--ask", case["id"]]]

    def both(path, case, query):
        first, second = (call_json(command, query) for command in commands(path, case))
        assert first == second, (case["id"], query, first, second)
        return first

    def rejected(path, case, query, reason=""):
        for command in commands(path, case):
            result = subprocess.run(command, input=json.dumps(query), capture_output=True,
                                    text=True, timeout=90)
            assert result.returncode and reason in result.stderr and not result.stdout, result

    rows = inspect_role(lambda query: both(suite_path, plos, query), plos, "Data curation", 2, 1)
    # Two attributions are separately retrievable for Timur. The other returned
    # target has an active credit but its assertion never supplied an authorized range.
    passages = [r["excerpt"] for public, _ in rows.values() for r in public]
    assert len(rows) == 2 and len(passages) == 3 and passages.count(None) == 1
    rejected(suite_path, shared, {**base, "kind": "evidence"}, "ambiguous evidence id")

    for field in ("at", "policy"):
        for malformed in (True, "1", 1.0, -1, 1 << 63):
            rejected(suite_path, shared, {**base, field: malformed})

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "variant.json"

        def write(case):
            path.write_text(json.dumps({"version": 1, "cases": [case]}), encoding="utf-8")

        for section, field in (("steps", "withdraw"), ("policies", "hidden"),
                               ("policies", "hidden_excerpts")):
            ambiguous = deepcopy(shared)
            ambiguous[section][1][field] = ["7"]
            write(ambiguous)
            rejected(path, ambiguous, base, "ambiguous assertion")

        # Policy 3 is supplied before either guild assertion exists. A public
        # client cannot infer future restricted support from an affirmative receipt.
        later = deepcopy(shared)
        guild = later["steps"][0]["put"].pop()
        later["steps"] = [later["steps"][0], {"revision": 2, "put": [guild]}]
        write(later)
        for at in (1, 2):
            visible = both(path, later, {**base, "policy": 3, "at": at})
            assert [(r["issuer"], r["id"]) for r in visible] == [("press", "7")]
        assert both(path, later, {**base, "policy": 3, "issuer": "guild", "at": 2}) == []
        staff = both(path, later, {**base, "policy": 3, "audience": "staff", "at": 2})
        assert [r["issuer"] for r in staff] == ["guild", "press"]
        # A bare policy selector stays ambiguous even when its local ID had
        # exactly one owner at the requested earlier factual edition.
        later["policies"][2]["hidden"] = ["7"]
        write(later)
        rejected(path, later, {**base, "policy": 3}, "ambiguous assertion")

        # Change only private neighboring bytes in separate possible worlds.
        changed = deepcopy(shared)
        changed["sources"][0]["text"] += " This archival note is restricted."
        write(changed)
        public = both(suite_path, shared, {**base, "policy": 2})
        assert public == both(path, changed, {**base, "policy": 2})
        staff = {**base, "policy": 2, "audience": "staff"}
        old, new = both(suite_path, shared, staff), both(path, changed, staff)
        def digest(row):
            return row["excerpt"]["citation"]["edition"]["source_sha256"]
        assert old and len(old) == len(new) and all(
            digest(left) != digest(right) for left, right in zip(old, new))
    print("Both runtime readers followed 2 returned targets to 3 credits, reopened 2 staff "
          "receipts, rejected ambiguous selectors, and isolated future hides and private source changes")
