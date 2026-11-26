import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


def pairs(groups):
    return {
        pair for group in groups for pair in itertools.combinations(sorted(group), 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Replay LitBank, reconcile published scores, and enumerate mention and link errors."
    )
    parser.add_argument("--anno", required=True, type=Path)
    parser.add_argument(
        "--assets",
        required=True,
        type=Path,
        help="Directory with litbank-test.parquet and five litbank-<arm>.json files from coref_litbank.py",
    )
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.anno / "scripts/qa"))
    from coref_litbank import index_partition, load, score_doc

    assets = args.assets
    docs = load(assets / "litbank-test.parquet")
    report = {
        "corpus_sha256": hashlib.sha256(
            (assets / "litbank-test.parquet").read_bytes()
        ).hexdigest(),
        "arms": {},
    }
    for model, mode in [
        ("mention-ranking", "predicted"),
        ("mention-ranking", "supplied"),
        ("fcoref", "predicted"),
        ("maverick", "predicted"),
        ("maverick", "supplied"),
    ]:
        name = "native" if model == "mention-ranking" else model
        arm = name + "-" + mode
        original = json.loads((assets / f"litbank-{arm}.json").read_text())
        inputs = [
            {
                "id": key,
                "text": text,
                "mentions": sorted(index_partition(gold))
                if mode == "supplied"
                else None,
            }
            for key, text, _, _, gold in docs
        ]
        run = subprocess.run(
            [str(args.driver.resolve()), model],
            check=False,
            input="".join(json.dumps(row) + "\n" for row in inputs),
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "ANNO_NO_DOWNLOADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        if run.returncode:
            raise RuntimeError(run.stderr[-1500:])
        totals, rows = Counter(), []
        for (key, text, _, _, gold), line, saved in zip(
            docs, run.stdout.splitlines(), original["documents"], strict=True
        ):
            record = json.loads(line)
            assert record["id"] == saved["id"] == key and record["producer"] == model
            row = {"id": key, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            assert row["text_sha256"] == saved["text_sha256"]
            rows.append(row)
            if "unsupported" in record:
                assert saved["unsupported"] == record["unsupported"]
                row["unsupported"] = record["unsupported"]
                continue
            pred = [set(map(tuple, group)) for group in record["clusters"]]
            assert score_doc(gold, pred) == saved["counts"]
            gold_mentions, predicted_mentions = (
                index_partition(gold),
                index_partition(pred),
            )
            gold_pairs, predicted_pairs = pairs(gold), pairs(pred)
            stats = Counter(
                matched_pairs=len(gold_pairs & predicted_pairs),
                missed_pairs=len(gold_pairs - predicted_pairs),
                spurious_pairs=len(predicted_pairs - gold_pairs),
                matched_mentions=len(gold_mentions.keys() & predicted_mentions.keys()),
                missed_mentions=len(gold_mentions.keys() - predicted_mentions.keys()),
                spurious_mentions=len(predicted_mentions.keys() - gold_mentions.keys()),
            )
            for a, b in gold_pairs - predicted_pairs:
                stats[
                    "missed_one_or_both_spans_absent"
                    if a not in predicted_mentions or b not in predicted_mentions
                    else "missed_both_spans_present_split"
                ] += 1
            for a, b in predicted_pairs - gold_pairs:
                stats[
                    "spurious_one_or_both_spans_unannotated"
                    if a not in gold_mentions or b not in gold_mentions
                    else "spurious_both_annotated_different_clusters"
                ] += 1
            totals.update(stats)
            row["counts"] = dict(stats)
            row["missed_pairs"] = sorted(gold_pairs - predicted_pairs)
            row["spurious_pairs"] = sorted(predicted_pairs - gold_pairs)
            row["missed_mentions"] = sorted(
                gold_mentions.keys() - predicted_mentions.keys()
            )
            row["spurious_mentions"] = sorted(
                predicted_mentions.keys() - gold_mentions.keys()
            )
        report["arms"][arm] = {"counts": dict(totals), "documents": rows}
        print(arm, dict(totals), flush=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
