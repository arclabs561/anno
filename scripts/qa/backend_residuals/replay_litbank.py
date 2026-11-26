import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/qa"))
from coref_gap import model_identity
from coref_litbank import index_partition, load, score_doc


def pairs(groups):
    return {
        pair for group in groups for pair in itertools.combinations(sorted(group), 2)
    }


def published_match(model, candidate, published):
    # The historical receipt identifies the source archive. The current worker
    # uses digest-verified safetensors prepared from that archive.
    identity = dict(candidate)
    if model == "maverick" and "source_sha256" not in published:
        identity["sha256"] = identity.pop("source_sha256")
    if identity != published:
        raise ValueError(f"{model} assets differ from the published panel")


def checked_partition(record, expected, length, model, mode):
    if "unsupported" in record or "unsupported" in expected:
        refusal = {"unsupported": "supplied_span_splits_token"}
        if ((model, mode) != ("maverick", "supplied") or expected != refusal
                or "clusters" in record or record.get("unsupported") != refusal["unsupported"]):
            raise ValueError("unexpected or ambiguous coreference refusal")
        return None
    groups = record["clusters"]
    if not isinstance(groups, list) or any(not isinstance(c, list) or not c for c in groups):
        raise ValueError("invalid coreference partition")
    pred = [[tuple(span) for span in group] for group in groups]
    for start, end in index_partition(pred):
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
            raise ValueError("invalid predicted source span")
    return [set(group) for group in pred]


def main():
    parser = argparse.ArgumentParser(
        description="Replay LitBank, reconcile published scores, and enumerate mention and link errors."
    )
    parser.add_argument(
        "--assets",
        required=True,
        type=Path,
        help="Directory with the pinned litbank-test.parquet",
    )
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local-details", type=Path, help="optional row-level ledger outside the checkout")
    args = parser.parse_args()
    if args.local_details and (args.local_details.resolve().is_relative_to(ROOT)
                               or args.local_details.resolve() == args.output.resolve()):
        parser.error("local details require a separate file outside the repository")
    published = json.loads((ROOT / "docs/eval/coref-panels.json").read_text())
    official = published["litbank"]

    assets = args.assets
    docs = load(assets / "litbank-test.parquet")
    report = {
        "corpus_sha256": hashlib.sha256(
            (assets / "litbank-test.parquet").read_bytes()
        ).hexdigest(),
        "arms": {},
    }
    with args.driver.open("rb") as driver:
        report["driver_sha256"] = hashlib.file_digest(driver, "sha256").hexdigest()
    local = {}
    for model, mode in [
        ("mention-ranking", "predicted"),
        ("mention-ranking", "supplied"),
        ("fcoref", "predicted"),
        ("maverick", "predicted"),
        ("maverick", "supplied"),
    ]:
        name = "native" if model == "mention-ranking" else model
        arm = name + "-" + mode
        identity = model_identity(model, mode)
        published_match(model, identity, published["models"][model])
        original = official["arms"][f"{model}/{mode}"]["per_id"]
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
            raise RuntimeError(f"{arm} driver exited with status {run.returncode}")
        totals, rows = Counter(), []
        for (key, text, _, _, gold), line, doc, saved in zip(
            docs, run.stdout.splitlines(), official["documents"], original, strict=True
        ):
            record = json.loads(line)
            if record["id"] != key or doc["id"] != key or record["producer"] != model:
                raise ValueError("wrong document or producer")
            row = {"id": key, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if row["text_sha256"] != doc["text_sha256"]:
                raise ValueError("source edition differs from published panel")
            rows.append(row)
            pred = checked_partition(record, saved, len(text), model, mode)
            if pred is None:
                row["unsupported"] = record["unsupported"]
                continue
            if score_doc(gold, pred) != saved:
                raise ValueError(f"{arm} scores differ from the published panel on {key}")
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
        refused = sum("unsupported" in row for row in rows)
        report["arms"][arm] = {
            "counts": dict(totals), "scored_documents": len(rows) - refused, "refused_documents": refused,
            "ordered_residuals_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "model": identity,
        }
        local[arm] = rows
        print(arm, dict(totals), flush=True)
    if args.local_details:
        args.local_details.write_text(json.dumps(local, separators=(",", ":")) + "\n")
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
