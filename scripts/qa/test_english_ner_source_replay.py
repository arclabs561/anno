"""Regression checks for source identity, raw review costs and paired NER scores."""

import json
import unittest
from copy import deepcopy

import english_ner_source_replay as study


class EnglishNerSourceTest(unittest.TestCase):
    def test_indexed_entities_unicode_and_repeated_text(self):
        source = (
            "#doc 11\n0\tÉva\tI-Peop\t['N']\t[0]\n1\t🚋\tO\t['N']\t[1]\n"
            "#doc 12\n0\tÉva\tO\t['N']\t[0]\n1\t🚋\tO\t['N']\t[1]\n"
        )
        cases, labels, tokens, tagged = study.cases_for("CoNLL04RE", source.encode())
        self.assertEqual(
            cases, [("11", "Éva 🚋", [(0, 3, "PER")]), ("12", "Éva 🚋", [])]
        )
        self.assertEqual((labels, tokens, tagged), ({"Peop": 1}, 4, 1))
        for wrong in (
            source.replace("#doc 12", "#doc 11"),
            source.replace("1\t🚋", "9\t🚋"),
        ):
            with self.assertRaises(ValueError):
                study.cases_for("CoNLL04RE", wrong.encode())
        sec = b"-DOCSTART- -X- O O\n\nAcme X - I-ORG\nInc X - I-ORG\n\n\nAcme X - O\n"
        self.assertEqual(
            study.cases_for("SECFilingsNER", sec)[0],
            [("FIN3-1", "Acme Inc", [(0, 8, "ORG")]), ("FIN3-2", "Acme", [])],
        )

    def test_alias_duplicate_and_ordinal_identity(self):
        dataset, backend, text = "SECFilingsNER", "bert-onnx", "é"
        digest = study.sha(text.encode())
        cases = [("FIN3-1", text, [(0, 1, "MISC")])]
        native = {
            "pairing": {
                "document_id": "SECFilingsNER:1",
                "source_sha256": digest,
                "gold_keys": {"0": study.native_gold_key("MISC", 0, 1)},
            },
            "model": backend,
            "gold_count": 1,
            "predicted_count": 3,
            "matches": [],
            "correct": 0,
        }
        pred = {
            "id": "FIN3-1",
            "model": backend,
            "runtime_model": backend,
            "source_sha256": digest,
            "text_length": 1,
            "entity_count": 3,
            "entities": [
                {"type": label, "text": text, "start": 0, "end": 1}
                for label in ("misc", "misc", "ORG")
            ],
        }
        report, hits = study.score(dataset, backend, cases, [native], [pred])
        self.assertEqual(hits, [{(0, 1, "MISC")}])
        self.assertEqual(report["literal_native_matches"], 0)
        self.assertEqual(report["all_labels"], study.metric(1, 3, 1))
        self.assertEqual(
            report["deduplicated_typed_predictions_counterfactual"],
            study.metric(1, 2, 1),
        )
        self.assertEqual(report["distinct_spans_with_competing_types"], 1)
        self.assertEqual(report["by_label"]["ORG"]["recall"], None)
        for field in ("document_id", "source_sha256", "gold_keys"):
            swapped = deepcopy(native)
            swapped["pairing"][field] = "swapped"
            with self.assertRaisesRegex(ValueError, "native reader"):
                study.score(dataset, backend, cases, [swapped], [pred])

    def test_published_receipts_reconcile_with_sources_and_costs(self):
        receipt = json.loads(
            (study.EVAL / "english-ner-source-replay.json").read_text()
        )
        chart = json.loads((study.EVAL / "english-ner-source-paired.json").read_text())
        self.assertEqual(study.paired(receipt), chart)
        for row in chart["rows"]:
            name = row["label"]
            source = study.source_for(name)
            result = receipt["datasets"][name]
            self.assertEqual(result["source"]["sha256"], source["sha256"])
            self.assertEqual(result["tokens"], int(source["tokens"]))
            self.assertEqual(result["tagged_tokens"], int(source["tagged"]))
            self.assertEqual(
                sum(result["observed_gold_labels"].values()), result["gold"]
            )
            for backend, field in zip(
                study.BACKENDS, ("baseline", "candidate"), strict=True
            ):
                metrics = result["backends"][backend]
                self.assertEqual(
                    sum(row[field].values()), metrics["all_labels"]["predicted"]
                )
                self.assertEqual(metrics["all_labels"]["gold"], result["gold"])
                dedup = metrics["deduplicated_typed_predictions_counterfactual"]
                self.assertEqual(dedup["correct"], row[field]["correct"])
                self.assertEqual(
                    dedup["predicted"] + metrics["repeated_typed_predictions"],
                    metrics["all_labels"]["predicted"],
                )
            self.assertEqual(
                row["gained"] - row["lost"],
                row["candidate"]["correct"] - row["baseline"]["correct"],
            )
        news, sec = (receipt["datasets"][n] for n in ("CoNLL04RE", "SECFilingsNER"))
        self.assertEqual(
            (news["cases"], news["gold"], news["gold_empty_cases"]), (288, 1079, 0)
        )
        self.assertEqual(
            (sec["cases"], sec["gold"], sec["gold_empty_cases"]), (303, 318, 203)
        )
        self.assertEqual(news["source_label_mapping"]["Other"], "Other")
        self.assertEqual(
            news["backends"]["bert-onnx"]["by_label"]["Other"]["correct"], 0
        )
        roles = sec["person_role_audit"]
        self.assertEqual(
            sum(r["exact"]["gold"] for r in roles.values()),
            sec["observed_gold_labels"]["PER"],
        )
        for r in roles.values():
            self.assertEqual(
                sum(r["candidate_on_lost_baseline_mentions"].values()),
                r["exact"]["lost"],
            )
            bounds = r["exact_boundary_any_type"]
            self.assertEqual(
                bounds["candidate"] - bounds["baseline"],
                bounds["gained"] - bounds["lost"],
            )
            for arm in ("baseline", "candidate"):
                self.assertGreaterEqual(bounds[arm], r["exact"][arm])
        self.assertEqual(roles["lender"]["exact_boundary_any_type"]["candidate"], 59)
        self.assertEqual(
            roles["borrower"]["exact"]["lost"] + roles["lender"]["exact"]["lost"], 161
        )


if __name__ == "__main__":
    unittest.main()
