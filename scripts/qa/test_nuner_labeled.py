"""Synthetic regressions for the dataset-independent NuNER scoring and public summary."""

import json
import unittest
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from audit_nuner_spans import predictions, run as audit_spans
    from nuner_error_analysis import checked_errors, public_summary
    from nuner_labeled import (
        CHECKPOINT, CHECKPOINT_REVISION, case_files_digest, check_graph_identity, check_run_artifacts,
        decode, digest, metrics, paired_interval, private_path,
    )


@unittest.skipIf(np is None, "NumPy is installed in the reference checks")
class NuNERScoringTests(unittest.TestCase):
    def test_private_paths_reject_checkout_and_symlinks_before_writing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "checkout"
            source.mkdir()
            alias = root / "alias"
            alias.symlink_to(source, target_is_directory=True)
            for path in (source / "panel.json", alias / "report.json"):
                with self.assertRaisesRegex(ValueError, "outside the Anno checkout"):
                    private_path(path, source)
            self.assertEqual(private_path(root / "private", source), root / "private")

    def test_changed_input_or_score_cannot_be_misattributed_to_run(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            panel, feed, scores = [work / name for name in ("panel.json", "input-0.npz", "u8-0.npy")]
            for path in (panel, feed, scores):
                path.write_bytes(b"original")
            stats = {"artifacts_sha256": {"panel": digest(panel), "ordered_feeds": case_files_digest([feed]),
                                           "ordered_logits": case_files_digest([scores])}}
            check_run_artifacts(work, "u8", stats, 1)
            for path in (panel, feed, scores):
                path.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "matched to this run"):
                    check_run_artifacts(work, "u8", stats, 1)
                path.write_bytes(b"original")

    def test_swapped_graph_cannot_be_reported_as_reference(self):
        fp32 = {"sha256": "001", "bytes": 12}
        u8 = {"sha256": "002", "bytes": 6}
        parity = {"fixture_checkpoint": f"{CHECKPOINT}@{CHECKPOINT_REVISION}", "fp32": fp32, "candidate": u8}
        check_graph_identity(parity, "fp32", fp32)
        with self.assertRaisesRegex(ValueError, "graph differs"):
            check_graph_identity(parity, "fp32", u8)

    def test_first_tie_and_inclusive_fp32_margin(self):
        words = [[0, 1], [2, 3], [4, 5]]
        scores = np.array([[30, 40, -20], [-10, -1e-8, -20], [-10, -1e-5, -20]], dtype=np.float32)
        expected = {(0, 1, 0), (2, 3, 1)}
        self.assertEqual(set(map(tuple, decode(scores, words, "A B C", 0.5))), expected)
        self.assertEqual(predictions(scores, words, "A B C", 0.5), expected)

    def test_paired_audit_rejects_count_preserving_receipt_changes(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows = [{"tokens": ["Åda", "Éva"], "ner_tags": [1, 0]},
                    {"tokens": ["Bea", "Lee"], "ner_tags": [0, 1]}]
            source = work / "test_en.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cases, saved_cases = [], []
            for i, row in enumerate(rows):
                case = {"group": "toy", "language": "en", "test_rows_zero_based": [i],
                        "words": [[0, 3], [4, 7]]}
                cases.append(case)
                scores = np.array([[4, -0.3, -3], [-2, -2, -2]], dtype=np.float32)
                np.save(work / f"fp32-{i}.npy", scores)
                if i == 0:
                    # A nonwinning label reaches exactly 0.5. The emitted span stays put.
                    scores[0, 1] = 0
                np.save(work / f"u8-{i}.npy", scores)
                saved_cases.append({
                    **case, "text_sha256": sha256(" ".join(row["tokens"]).encode()).hexdigest(),
                    "gold": [[0, 3, 0]] if i == 0 else [[4, 7, 0]],
                    "fp32": [[0, 3, 0]], "u8": [[0, 3, 0]],
                    "threshold_crossings": {"0.3": 0, "0.5": int(i == 0), "0.7": 0},
                })
            (work / "panel.json").write_text(json.dumps(cases))
            counts = {"tp": 1, "fp": 1, "fn": 1}
            group = {"n": 2, "thresholds": {key: counts for key in ("0.3", "0.5", "0.7")}}
            receipt = {"test_file_sha256": {"en": digest(source)}, "cases": saved_cases,
                       "labels_by_index": ["person", "organization", "location"],
                       "models": {arm: {"groups": {"toy": group}} for arm in ("fp32", "u8")}}
            self.assertEqual(audit_spans(work, receipt), {arm: {"toy": counts} for arm in ("fp32", "u8")})

            # Assign the same total number of crossings to the wrong case.
            altered = deepcopy(receipt)
            altered["cases"][0]["threshold_crossings"]["0.5"] = 0
            altered["cases"][1]["threshold_crossings"]["0.5"] = 1
            with self.assertRaisesRegex(ValueError, "threshold crossings differ on case 0"):
                audit_spans(work, altered)

            # A different false-positive boundary or type leaves both arms' TP/FP/FN unchanged.
            for replacement in ([1, 3, 0], [0, 3, 1]):
                with self.subTest(replacement=replacement):
                    altered = deepcopy(receipt)
                    altered["cases"][1]["u8"] = [replacement]
                    with self.assertRaisesRegex(ValueError, "u8: different span inventory on case 1"):
                        audit_spans(work, altered)

    def test_undefined_micro_f1(self):
        interval = paired_interval([{"gold": [], "fp32": [], "u8": []}])
        self.assertEqual(interval["defined"], 0)
        self.assertIsNone(interval["percentile_95"])
        self.assertIsNone(metrics(0, 0, 0)["f1"])

    def test_empty_case_is_part_of_sample_and_summary_excludes_private_fields(self):
        case = {"gold": [], "fp32": [], "u8": [], "group": "en-random", "compared_word_labels": 3,
                "max_absolute_probability_change": 0.0, "threshold_crossings": {"0.3": 0, "0.5": 0, "0.7": 0}}
        matched = {**case, **{name: [[0, 1, 0]] for name in ("gold", "fp32", "u8")}}
        group = {"n": 2, "elapsed_seconds": 1.0, "median_seconds": 0.5,
                 "thresholds": {"0.5": metrics(1, 1, 1)}, "private_words": ["sentinel"]}
        stats = {k: 0 for k in ("sha256", "bytes", "ort", "provider", "machine", "cpu_affinity", "intra_threads",
                                 "inter_threads", "session_load_seconds", "process_peak_rss_kib")}
        stats.update(groups={"en-random": group}, timings_seconds=[0.5, 0.5])
        ledger = {k: {} for k in ("checkpoint", "checkpoint_revision", "dataset", "dataset_revision",
                                  "test_file_sha256", "labels_by_index", "packages", "paired_f1_delta_u8_minus_fp32_at_0.5")}
        ledger.update(cases=[case, matched], models={key: deepcopy(stats) for key in ("fp32", "u8")})
        public = public_summary(ledger, checked_errors(ledger))
        self.assertEqual(public["comparison"]["cases"], 2)
        self.assertNotIn("cases", public)
        self.assertNotIn("timings_seconds", public["models"]["fp32"])
        self.assertNotIn("private_words", public["models"]["fp32"]["groups"]["en-random"])
        ledger["cases"].pop(0)
        with self.assertRaisesRegex(ValueError, "case counts"):
            checked_errors(ledger)


if __name__ == "__main__":
    if np is None:
        raise SystemExit("Install numpy to run the NuNER scoring checks")
    unittest.main()
