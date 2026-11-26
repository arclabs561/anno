import io
import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest

from contextlib import redirect_stderr, redirect_stdout

from replay_matrix_feedback import sampler_env
from check_matrix_outcomes import RANDOM_TASKS, WIKIGOLD, check, check_fixed_sample, check_temporal_contrast, check_temporal_smoke, check_ner_smoke, check_wikigold_bootstrap, report_dataset_pools


def outcome(task, strategy="random", hard=False, examples=1):
    return {
        "record_type": "outcome",
        "strategy": strategy,
        "slice": f"{task}.lang=en",
        "hard_junk": hard,
        "cost_units": examples,
        "primary_f1": 0,
        "ok": False,  # A valid zero score counts as an execution.
    }


class CoverageTest(unittest.TestCase):
    def accepted(self, rows):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return check(map(json.dumps, rows))

    def test_replay_preserves_build_revision_while_isolating_sampler_settings(self):
        root = Path("/example")
        with patch.dict(os.environ, {"ANNO_GIT_COMMIT": "a" * 40, "ANNO_WORST_SOFT_WEIGHT": "99"}, clear=True):
            env = sampler_env(root, "random", root / "rows.jsonl")
            self.assertEqual(env["ANNO_GIT_COMMIT"], "a" * 40)
            self.assertEqual(env["ANNO_WORST_SOFT_WEIGHT"], "1")
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("ANNO_GIT_COMMIT", sampler_env(root, "random", root / "rows.jsonl"))

    def test_zero_scores_and_a_failed_worst_first_attempt(self):
        rows = [outcome(task) for task in RANDOM_TASKS]
        rows.append(outcome("ner", "worst-first", hard=True, examples=0))
        self.assertTrue(self.accepted(rows))

    def test_repeated_ner_cannot_cover_other_tasks(self):
        rows = [outcome("ner")] * 9 + [outcome("ner", "worst-first")] * 2
        self.assertFalse(self.accepted(rows))

    def test_hard_failures_and_empty_samples_do_not_cover_a_task(self):
        rows = [outcome(task) for task in RANDOM_TASKS if task != "re"]
        rows.extend([outcome("re", hard=True), outcome("re", examples=0)])
        rows.append(outcome("re", "worst-first"))
        self.assertFalse(self.accepted(rows))

    def test_language_report_counts_observed_nonempty_outcomes_only(self):
        rows = [outcome("ner"), dict(outcome("ner"), observed_language="de"),
                dict(outcome("ner", hard=True), observed_language="fr"),
                dict(outcome("ner", examples=0), observed_language="en")]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            check(map(json.dumps, rows), report=True)
        self.assertIn("| ner | unverified | 1 |", output.getvalue())
        self.assertIn("| ner | de | 1 |", output.getvalue())
        self.assertNotIn("| en |", output.getvalue())
        self.assertNotIn("| fr |", output.getvalue())

    def test_policy_decision_alone_does_not_count_as_an_outcome(self):
        rows = [outcome(task) for task in RANDOM_TASKS]
        rows.append({"strategy": "worst-first", "chosen": "tplinker"})
        self.assertFalse(self.accepted(rows))

    def test_invalid_log_identifies_its_line(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            check([json.dumps(outcome("re")), "{"])

    def test_empty_cache_is_visible_in_exploratory_report(self):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertTrue(check([], report=True))
        self.assertIn("temporal (no outcome)", output.getvalue())

    def test_dataset_pools_separate_reader_failures_and_unexamined_candidates(self):
        pool = {"catalog_count": 6, "eligible_count": 4, "excluded": {"cache_required": 2},
                "selected": ["AnnoTimexSmoke"], "backend_incompatible_count": 1,
                "load_rejections": {"dataset parse error": 1}}
        row = {"schema_version": 1, "strategy": "random", "slice": "temporal",
               "pass": "initial", "pool": pool}
        output = io.StringIO()
        with redirect_stdout(output):
            report_dataset_pools([json.dumps(row)])
        self.assertIn("4/6 policy-eligible, 2 load checks, 1 selected, 1 incompatible, 1 unexamined", output.getvalue())
        self.assertIn("Load rejections: dataset parse error (1)", output.getvalue())
        example = {"dataset": "CorpusB", "reason": "dataset parse error"}
        with redirect_stdout(output):
            report_dataset_pools([json.dumps(row | {"pool": pool | {"rejection_examples": [example]}})])
        self.assertIn("Rejected CorpusB (dataset parse error).", output.getvalue())
        for change in ({"catalog_count": 5}, {"selected": "unparsed"}, {"backend_incompatible_count": -1}):
            with self.assertRaisesRegex(ValueError, "line 1"):
                report_dataset_pools([json.dumps(row | {"pool": pool | change})])

    def test_matching_digests_report_distinct_dataset_ids_once(self):
        digest = "a" * 64
        rows = [dict(outcome("ner"), dataset=ds, dataset_sha256=sha)
                for ds, sha in (("CorpusA", digest), ("CorpusA", digest), ("CorpusB", digest),
                                ("MissingHash", None), ("ShortHash", "a"), ("Other", "b" * 64))]
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            check(map(json.dumps, rows), report=True)
        matches = [line for line in output.getvalue().splitlines() if "share recorded input SHA-256" in line]
        self.assertEqual(matches, [f"Dataset IDs CorpusA, CorpusB share recorded input SHA-256 `{digest}`."])

    def test_failed_cells_group_retries_and_keep_tasks_distinct(self):
        row = dict(outcome("ner", hard=True), dataset="WikiGold", backend="gliner_relex",
                   fail_kind="backend")
        retry = dict(row, strategy="worst-first")
        relation = dict(row, slice="re.lang=en")
        success = dict(row, hard_junk=False, backend="heuristic")
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            check(map(json.dumps, [row, retry, relation, success]), report=True)
        self.assertIn("ner / WikiGold / gliner_relex: backend (2)", output.getvalue())
        self.assertIn("re / WikiGold / gliner_relex: backend (1)", output.getvalue())
        self.assertNotIn("heuristic", output.getvalue())

    def test_resolved_replay_uses_requested_limit_and_loader_digest(self):
        row = dict(outcome("ner", examples=1), dataset="AnnoNerSmoke", backend="heuristic",
                   task="ner", max_examples=5, seed=913, dataset_sha256="a" * 64,
                   coref_gold=False, coref_english_pronouns=False)
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(io.StringIO()):
            self.assertTrue(check(map(json.dumps, [row, row]), report=True))
        cmd = "python3 scripts/qa/replay_matrix_cell.py ner AnnoNerSmoke heuristic"
        self.assertEqual(output.getvalue().count(cmd), 1)
        self.assertIn(cmd + " --seed 913 --max-examples 5 --dataset-sha256 " + "a" * 64,
                      output.getvalue())

    def test_language_replay_requires_safe_tags_and_preserves_pins(self):
        from replay_matrix_cell import replay_command
        row = dict(outcome("ner"), task="ner", dataset="WikiANN", backend="heuristic",
                   max_examples=4, seed=5, dataset_sha256="a" * 64, coref_gold=False,
                   coref_english_pronouns=False, requested_languages=["de", "fr"], observed_language="fr")
        self.assertIn("--languages de,fr --observed-language fr", replay_command(row))
        for bad in ("xx;echo", ["de", "de"], [1], ["invalid lang"]):
            self.assertIsNone(replay_command(row | {"requested_languages": bad}))
        self.assertIsNone(replay_command(row | {"observed_language": "de\nfr"}))

    def test_legacy_and_untrusted_fields_cannot_form_a_command(self):
        row = dict(outcome("ner"), task="ner", dataset="WikiGold", backend="heuristic",
                   max_examples=5, seed=42, coref_gold=False, coref_english_pronouns=False)
        for change in ({"task": None}, {"dataset": "../foreign"}, {"backend": "bad;command"},
                       {"seed": True}, {"max_examples": -1}, {"dataset_sha256": "bad`receipt"},
                       {"coref_gold": 1}):
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                self.assertTrue(check([json.dumps(row | change)], report=True))
            self.assertNotIn("replay_matrix_cell.py", output.getvalue())

    def test_smoke_requires_exact_fixture_result(self):
        row = dict(outcome("ner"), backend="heuristic", dataset="AnnoNerSmoke", primary_f1=1.0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(check_ner_smoke([json.dumps(row)]))
            self.assertFalse(check_ner_smoke([]))
            for mutation in ({"primary_f1": 0.0}, {"backend": "crf"}, {"cost_units": 0}, {"dataset": "WikiGold"}):
                self.assertFalse(check_ner_smoke([json.dumps(row | mutation)]))

    def test_wikigold_bootstrap_requires_real_dataset_identity(self):
        row = dict(outcome("ner", examples=5), backend="heuristic", dataset="WikiGold", dataset_sha256=WIKIGOLD)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(check_wikigold_bootstrap([json.dumps(row)]))
            self.assertFalse(check_wikigold_bootstrap([json.dumps(row | {"dataset": "AnnoNerSmoke"})]))
            for digest in (None, "0" * 64):
                self.assertFalse(check_wikigold_bootstrap([json.dumps(row | {"dataset_sha256": digest})]))

    def test_prepared_wikigold_digest_required_even_in_advisory_report(self):
        row = dict(outcome("ner"), dataset="WikiGold")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for digest in (None, "0" * 64, WIKIGOLD):
                self.assertEqual(
                    check([json.dumps(row | {"dataset_sha256": digest})], report=True, verify_wikigold=True),
                    digest == WIKIGOLD,
                )
            self.assertTrue(check([json.dumps(row)], report=True))
            self.assertTrue(check([], report=True, verify_wikigold=True))
            self.assertTrue(check([json.dumps(row | {"hard_junk": True, "cost_units": 0})],
                                  report=True, verify_wikigold=True))

    def test_public_timex_requires_a_valid_f1_over_the_whole_split(self):
        dataset = "TimexRecognitionSentenceOriginal"
        row = dict(outcome("temporal", examples=390), backend="pattern", dataset=dataset)
        def accepts(**changes):
            return check_fixed_sample([json.dumps(row | changes)], task="temporal", dataset=dataset,
                                      backend="pattern", examples=390, exact=False)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            for score in (0, 0.10698, 1):
                self.assertTrue(accepts(primary_f1=score))
            for score in (-0.01, 4/3, float("nan"), float("inf"), True):
                self.assertFalse(accepts(primary_f1=score))
            self.assertFalse(accepts(cost_units=389))
            self.assertFalse(accepts(hard_junk=True))

    def test_contrast_control_requires_the_whole_offline_panel(self):
        row = dict(outcome("temporal", examples=34), dataset="AnnoTimexContrast",
                   backend="pattern", primary_f1=0.4)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(check_temporal_contrast([json.dumps(row)]))
            for change in ({"cost_units": 390}, {"cost_units": 2}, {"hard_junk": True},
                           {"dataset": "TimexRecognitionSentenceOriginal"}, {"dataset": "AnnoTimexSmoke"}):
                self.assertFalse(check_temporal_contrast([json.dumps(row | change)]))

    def test_temporal_fixture_requires_the_positive_and_negative_documents(self):
        row = dict(outcome("temporal", examples=2), dataset="AnnoTimexSmoke",
                   backend="pattern", primary_f1=1.0,
                   structural_counts={"gold_spans": 1, "predicted_spans": 1, "matched_spans": 1,
                                      "gold_empty_sentences": 1, "gold_empty_sentences_with_predictions": 0,
                                      "unmatched_predictions_overlapping_missed_gold": 0})
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertTrue(check_temporal_smoke([json.dumps(row)]))
            for change in ({"structural_counts": {}}, {"structural_counts": None},
                           {"structural_counts": row["structural_counts"] | {"matched_spans": True}},
                           {"cost_units": 1}, {"primary_f1": 0.8}, {"hard_junk": True},
                           {"slice": "ner.lang=en"}, {"dataset": "WikiGold"},
                           {"dataset": "TimexRecognitionSentenceOriginal"}, {"dataset": "AnnoTimexContrast"}):
                self.assertFalse(check_temporal_smoke([json.dumps(row | change)]))

    def test_reports_first_decision_and_matching_observation(self):
        decision = {"strategy": "worst-first", "chosen": "heuristic", "slice": "ner",
                    "run_id": "run", "datasets": ["WikiGold"], "explore_first": False,
                    "top_candidates": {"rows": [{"arm": "heuristic", "calls": 2}]}}
        decision["worst_first_round"] = {"retry": {"observation_id": "prior", "reason": "recurrent_after_recovery", "risk": 0.25, "observations": 4}}
        wrong = dict(outcome("ner", "worst-first"), backend="crf", dataset="WikiGold", run_id="run")
        matching = dict(wrong, backend="heuristic", hard_junk=True, fail_kind="io")
        for rows, result in (([decision, wrong], "no outcome"), ([decision, wrong, matching], "hard failure")):
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(io.StringIO()):
                check(map(json.dumps, rows), report=True)
            if matching in rows:
                self.assertIn("Hard failures by recorded kind: io 1", output.getvalue())
            self.assertIn("Retry evidence: prior (risk weight 0.250 over 4 observations)", output.getvalue())
            self.assertIn(f"heuristic (repeated failures after recovery, prior calls 2) — {result}", output.getvalue())
