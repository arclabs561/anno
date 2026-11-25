# SPDX-License-Identifier: MIT OR Apache-2.0
"""Replay the caption example using a local FP32 export and compiled Anno binary."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from caption_links import map_clusters, transcript

HERE = Path(__file__).resolve().parent


class CaptionLinks(unittest.TestCase):
    binary = None
    model = None

    def test_model_round_trip(self):
        if self.binary is None:
            self.skipTest("run this script with a built fcoref_word_spans binary and FP32 export")
        original = json.loads((HERE / "gate-captions.json").read_text(encoding="utf-8"))
        for prefix in ("", "🎙 "):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                cues = [dict(cue) for cue in original]
                cues[0]["text"] = prefix + cues[0]["text"]
                path = Path(tmp) / "cues.json"
                path.write_text(json.dumps(cues), encoding="utf-8")
                output = subprocess.check_output(
                    [sys.executable, HERE / "caption_links.py", self.binary, self.model, path]
                )
                receipt = json.loads(output)
                joined = "\n".join(c["text"] for c in cues)
                self.assertEqual(receipt["text_sha256"], hashlib.sha256(joined.encode()).hexdigest())
                clusters = receipt["suggested_clusters"]
                for left, right in [("Émilie Durand", "She"), ("the east gate", "it")]:
                    self.assertTrue(any({left, right} <= {m["quote"] for m in c} for c in clusters))
                mention, = [m for c in clusters for m in c if m["quote"] == "She"]
                self.assertEqual(mention["chars"], [39 + len(prefix), 42 + len(prefix)])
                self.assertEqual(mention["cues"], [
                    {"id": "cue-18", "start_ms": 14900, "end_ms": 17600, "chars": [0, 3]}
                ])
                by_id = {c["id"]: c for c in cues}
                bases = {"cue-17": 0, "cue-18": len(cues[0]["text"]) + 1}
                for member in (m for cluster in clusters for m in cluster):
                    start, end = member["chars"]
                    self.assertEqual(joined[start:end], member["quote"])
                    self.assertTrue(member["cues"])
                    for hit in member["cues"]:
                        cue = by_id[hit["id"]]
                        self.assertEqual((hit["start_ms"], hit["end_ms"]), (cue["start_ms"], cue["end_ms"]))
                        a, b = hit["chars"]
                        base = bases[hit["id"]]
                        self.assertEqual((base + a, base + b),
                                         (max(start, base), min(end, base + len(cue["text"]))))

    def test_cross_cue_quote_and_source_identity(self):
        cues = [
            {"id": "a", "start_ms": 0, "end_ms": 500, "text": "🎙 e\u0301"},
            {"id": "b", "start_ms": 700, "end_ms": 900, "text": "Nu"},
        ]
        text, positions = transcript(cues)
        result = {"id": "captions", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "clusters": [[[2, 7]]]}
        hit = map_clusters(text, positions, {"results": [result]})["suggested_clusters"][0][0]
        self.assertEqual(hit["quote"], "e\u0301\nNu")
        self.assertEqual(hit["cues"], [
            {"id": "a", "start_ms": 0, "end_ms": 500, "chars": [2, 4]},
            {"id": "b", "start_ms": 700, "end_ms": 900, "chars": [0, 2]},
        ])
        with self.assertRaisesRegex(ValueError, "receipt"):
            map_clusters(text + "!", positions, {"results": [result]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("model")
    args = parser.parse_args()
    CaptionLinks.binary, CaptionLinks.model = args.binary, args.model
    unittest.main(argv=[sys.argv[0]])
