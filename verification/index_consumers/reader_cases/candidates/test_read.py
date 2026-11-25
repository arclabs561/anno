# SPDX-License-Identifier: MIT OR Apache-2.0
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[4]
COMMAND = [
    sys.executable,
    "-m",
    "verification.index_consumers.reader_cases.candidates.read",
]


def run(request):
    return subprocess.run(
        COMMAND, input=json.dumps(request), text=True, cwd=ROOT, capture_output=True
    )


class ProposalCLI(unittest.TestCase):
    def test_original_spellings_separate_occurrences_and_mixed_snippet(self):
        native = dict(
            source="register", revision="r2", occurrence="n1", surface="Éra", group="a"
        )
        earlier = dict(native, revision="r1")
        first = dict(
            source="folio",
            revision="9",
            occurrence="w1",
            surface="ERA",
            group="b",
            snippet="ERA copied the chart. A second ERA annotated it under another byline.",
        )
        second = dict(first, occurrence="w2", group="c")
        req = {
            "query": {"surface": "Éra", "group": "a"},
            "variant": "fold",
            "gallery": [native, first, earlier],
        }
        result = run(req)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual([h["revision"] for h in data["native_hits"]], ["r1", "r2"])
        self.assertEqual(
            data["proposal"],
            {
                "tier": "fold",
                "decision": "suggested",
                "hits": [
                    {
                        "source": "folio",
                        "revision": "9",
                        "occurrence": "w1",
                        "surface": "ERA",
                    }
                ],
            },
        )
        self.assertNotIn("copied", result.stdout)
        req["gallery"] = [second, earlier, first, native]
        ambiguous = json.loads(run(req).stdout)
        self.assertEqual(
            ambiguous["proposal"], {"tier": "fold", "decision": "ambiguous", "hits": []}
        )
        self.assertEqual(ambiguous["native_hits"], data["native_hits"])
        req["gallery"].reverse()
        self.assertEqual(json.loads(run(req).stdout), ambiguous)

    def test_rejects_duplicates_and_malformed_input(self):
        row = dict(source="s", revision="1", occurrence="o", surface="Téra", group="g")
        req = {
            "query": {"surface": "Tera", "group": None},
            "variant": "expanded",
            "gallery": [row],
        }
        self.assertEqual(json.loads(run(req).stdout)["native_hits"], [])
        for bad in (
            [row, dict(row, surface="other")],
            [dict(row, group=None)],
            [dict(row, revision=1)],
        ):
            with self.subTest(gallery=bad):
                result = run({**req, "gallery": bad})
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
