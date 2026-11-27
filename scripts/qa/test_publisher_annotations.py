"""The source survey must not turn overlapping or incomplete labels into density."""

import hashlib
import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import survey_publisher_annotations as publisher
    import survey_scrolls_qmsum
    import survey_slue_manifest
finally:
    sys.path.pop(0)


class PublisherAnnotationsTest(unittest.TestCase):
    def test_nested_and_unmarked_nne_sentences(self):
        counts = publisher.nne("0 1 2\nNN NNP NNP\n1,2 PER|2,2 NAME\n\n0\nNN\n\n")
        self.assertEqual(counts["tokens"], 4)
        self.assertEqual(counts["tagged tokens"], 2)
        self.assertEqual(counts["mention annotations"], 2)
        with self.assertRaisesRegex(ValueError, "outside sentence"):
            publisher.nne("0 1\nNN NN\n1,2 PER\n")

    def test_open_markup_withholds_the_whole_file_density(self):
        good = "<mark>Red <mark>Elm</mark></mark>, spoke."
        counted = publisher.marked_sentences(good)
        self.assertEqual((counted["tokens"], counted["tagged tokens"]), (3, 2))
        self.assertEqual(counted["mention annotations"], 2)
        partial = publisher.marked_sentences(good + "\n<mark>broken span")
        self.assertEqual(partial["tokens"], 5)
        self.assertNotIn("tagged tokens", partial)
        self.assertEqual(partial["sentences with unbalanced markup"], 1)
        self.assertEqual(partial["tagged tokens in balanced sentences"], 2)

    def test_finnish_nested_columns_and_unlabeled_structural_rows(self):
        rows = "<HEADLINE>\t\t\nA\tB-ORG\tO\nB\tI-ORG\tB-PER\nword\tO\tO\n"
        counted = publisher.token_labels("FinNER", rows)
        self.assertEqual(
            counted, {"tokens": 3, "tagged tokens": 2, "mention annotations": 2}
        )
        partial = publisher.token_labels("MHERCL", "word\tQ\t_\tword\tNOUN")
        self.assertNotIn("tagged tokens", partial)
        self.assertEqual(partial["unrecognized BIO labels"], 1)

    def test_sanskrit_multiword_wrappers_do_not_inflate_tokens(self):
        rows = [
            "# sent_id = 1",
            "1-2\tword\t_\t_\t_\t_\t_\t_\t_\t_",
            "1\tw\t_\t_\t_\t_\t_\t_\t_\tEntity=(e1-person",
            "2\tx\t_\t_\t_\t_\t_\t_\t_\tEntity=e1)",
            "3\t.\t_\t_\t_\t_\t_\t_\t_\t_",
        ]
        counted = publisher.conllu("\n".join(rows), entity=True)
        self.assertEqual(
            (
                counted["tokens"],
                counted["tagged tokens"],
                counted["mention annotations"],
            ),
            (3, 2, 1),
        )

    def test_maven_unlabeled_test_triggers_cannot_become_gold_mentions(self):
        document = {
            "id": "1",
            "title": "Article",
            "content": [{"tokens": ["one", "two"]}],
            "candidates": [{"trigger_word": "two"}],
        }
        counts = publisher.maven_counts([json.dumps(document)], "MAVEN")
        self.assertEqual(
            counts, {"documents": 1, "tokens": 2, "unlabeled trigger candidates": 1}
        )

    def test_belb_sentinel_corrections_are_not_measured_spans(self):
        payload = [
            {"id": 1, "start_patch": 0, "end_patch": 2, "text_patch": "AB"},
            {"id": 2, "start_patch": -1, "end_patch": -1, "text_patch": "CD"},
        ]
        self.assertEqual(
            publisher.file_metrics("BELB", json.dumps(payload).encode()),
            {"mention annotations": 1, "corrections without replacement offsets": 1},
        )

    def test_qmsum_source_unit_is_a_meeting_with_multiple_queries(self):
        meeting = {
            "meeting_transcripts": [{"speaker": "A", "content": "Hello"}],
            "general_query_list": [{"query": "Summary?", "answer": "One greeting"}],
            "specific_query_list": [{"query": "Who spoke?", "answer": "A"}],
        }
        line = json.dumps(meeting).encode()
        expected = {"documents": 2, "questions": 4, "turns": 2}
        self.assertEqual(
            publisher.file_metrics("ScrollsQMSum", line + b"\n" + line), expected
        )

    def test_nonempty_scrolls_placeholder_is_not_a_verified_reference(self):
        source = '"two original summaries"'
        full = hashlib.sha256(
            json.dumps([source, source], separators=(",", ":")).encode()
        ).hexdigest()
        payload = [
            json.dumps({"id": str(i), "input": "input", "output": "placeholder"})
            for i in range(2)
        ]
        result = survey_scrolls_qmsum.partition_counts(payload, full)
        self.assertEqual(result["questions with nonempty output"], 2)
        self.assertEqual(result["distinct nonempty output texts"], 1)
        self.assertNotIn("reference summaries verified against original QMSum", result)
        actual = [
            json.dumps({"id": str(i), "input": "input", "output": source})
            for i in range(2)
        ]
        self.assertEqual(
            survey_scrolls_qmsum.partition_counts(actual, full)[
                "reference summaries verified against original QMSum"
            ],
            2,
        )

    def test_blind_slue_uses_only_manifest_byte_ranges_and_omits_test_labels(self):
        packed = io.BytesIO()
        with zipfile.ZipFile(packed, "w") as archive:
            archive.writestr("audio.ogg", b"a" * 90000)
            for split in survey_slue_manifest.SPLITS:
                rows = "id\tnormalized_text\tnormalized_ner\n1\tactor in Rome\t[['GPE', 9, 4]]\n"
                if split == "test_blind":
                    rows = "id\tnormalized_text\n1\tactor in Rome\n"
                archive.writestr(
                    "slue-voxpopuli/slue-voxpopuli_" + split + ".tsv", rows
                )
        data = packed.getvalue()
        served = []

        def fetch(request, timeout):
            lo, hi = map(
                int,
                re.fullmatch(
                    r"bytes=(\d+)-(\d+)", request.get_header("Range")
                ).groups(),
            )
            response = io.BytesIO(data[lo : hi + 1])
            response.status = 206
            response.headers = {
                "Content-Range": f"bytes {lo}-{hi}/{len(data)}",
                "ETag": '"pinned"',
            }
            served.append(hi - lo + 1)
            return response

        rows = survey_slue_manifest.survey({"id": "SLUE"}, fetch=fetch)
        measured = {(r["split"], r["unit"]): r["value"] for r in rows}
        self.assertEqual(measured["dev", "tagged tokens"], 1)
        self.assertEqual(measured["test_blind", "tokens"], 3)
        self.assertNotIn(("test_blind", "tagged tokens"), measured)
        self.assertLess(sum(served), 4096)

    def test_archive_reads_only_the_reviewed_annotation_layer(self):
        relative = "data/annotation/1.json"
        payload = json.dumps(
            {
                "sourcefile": "one.txt",
                "cyberevent": {
                    "hopper": [{"events": [{"nugget": "news", "type": "event"}]}]
                },
            }
        ).encode()
        base = {"id": "CASIE", "repository": "owner/repo", "revision": "1" * 40}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            with zipfile.ZipFile(cache / (base["revision"] + ".zip"), "w") as archive:
                archive.writestr("corpus/" + relative, payload)
                archive.writestr("corpus/data/source/1.json", payload)
            tree = {
                "files": {relative: len(payload), "data/source/1.json": len(payload)}
            }
            rows = publisher.archive_metrics(base, tree, cache)
            self.assertEqual(
                {r["unit"]: r["value"] for r in rows},
                {
                    "documents": 1,
                    "source members": 1,
                    "event nuggets": 1,
                    "annotated argument pairs": 0,
                },
            )
            tree["files"][relative] += 1
            self.assertEqual(
                publisher.archive_metrics(base, tree, cache)[0]["status"], "unmeasured"
            )


if __name__ == "__main__":
    unittest.main()
