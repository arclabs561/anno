# /// script
# requires-python = ">=3.10"
# dependencies = ["xxhash==3.6.0"]
# ///
"""Replay Anno's heuristic or its CLI default on every pinned WNUT-17 test sentence."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_wnut_novelty import GOLD, MANIFEST, REVISION, load_file, surface

BASELINE_REVISION = "8395db97f26d47bbe2cb9fe33473ddeaac699605"
CANDIDATE_REVISION = "f9427977f2f27b32f26c4c77a6795a8a4dce596c"
# Anno's social_media() mapper groups corporation and group; WNUT uses a hyphen for creative-work.
PROJECTION = {
    "person": "PER",
    "location": "LOC",
    "corporation": "ORG",
    "group": "ORG",
    "creative-work": "WORK_OF_ART",
    "product": "PRODUCT",
}
OUTCOMES = ("exact", "wrong_type", "partial_overlap", "no_overlap")
ARMS = ("baseline", "candidate")
RUNTIME = {"heuristic": "heuristic", "stacked": "stacked(regex+heuristic)"}
PRODUCERS = {"heuristic": {"heuristic"}, "stacked": {"pattern", "heuristic"}}


def sentence_cases(tagged):
    annotations = defaultdict(list)
    for index, start, end, kind in sorted(tagged.spans):
        annotations[index].append((start, end, kind))
    cases = []
    for index, tokens in enumerate(tagged.tokens):
        text = " ".join(tokens)
        starts = [0]
        for token in tokens:
            starts.append(starts[-1] + len(token) + 1)
        gold = {
            (starts[start], starts[end] - 1, kind)
            for start, end, kind in annotations[index]
        }
        cases.append((text, gold))
    return cases


def load_cases(data_dir):
    with MANIFEST.open(encoding="utf-8", newline="") as src:
        hashes = {
            r["path"]: r["sha256"]
            for r in csv.DictReader(src)
            if r["id"] == "Wnut17" and r["revision"] == REVISION and r["status"] == "ok"
        }
    splits = {
        s: load_file(GOLD[s], hashes[GOLD[s]], data_dir) for s in ("Train", "Test")
    }
    train, test = splits["Train"], splits["Test"]
    names = {surface(span, train.tokens).casefold() for span in train.spans}
    cases = sentence_cases(test)
    panel = hashlib.sha256()
    for text, _ in cases:
        data = text.encode("utf-8")
        panel.update(len(data).to_bytes(8, "big") + data)
    source = {
        "repository": "leondz/emerging_entities_17",
        "revision": REVISION,
        "sha256": {GOLD[s]: hashes[GOLD[s]] for s in splits},
        "train_sentences": len(train.tokens),
        "train_mentions": len(train.spans),
        "train_casefolded_names": len(names),
        "test_sentences": len(cases),
        "test_distinct_reconstructed_texts": len({t for t, _ in cases}),
        "test_gold_empty_sentences": sum(not gold for _, gold in cases),
        "test_tokens": sum(map(len, test.tokens)),
        "test_mentions": len(test.spans),
        "test_reconstructed_texts_sha256": panel.hexdigest(),
    }
    return source, cases, names


def check_prediction(record, text, text_hash, require_runtime, backend="heuristic"):
    prov = record["provenance"]
    runtime = prov.get("runtime_model")
    if prov["model"] != backend or (
        runtime != RUNTIME[backend] and (require_runtime or runtime is not None)
    ):
        raise ValueError("Unexpected requested or instantiated backend")
    rows = record["entities"]
    if (
        prov["text_hash"] != text_hash
        or prov["text_chars"] != len(text)
        or prov["entity_count"] != len(rows)
    ):
        raise ValueError(
            "Native input hash, length or output count disagrees with the replay"
        )
    spans = set()
    sources = Counter()
    for row in rows:
        start, end, label = row["start"], row["end"], row["type"]
        if not 0 <= start < end <= len(text) or text[start:end] != row["text"]:
            raise ValueError(
                "Native span does not slice its input by Unicode character offsets"
            )
        if not label or not 0.5 <= row["confidence"] <= 1.0:
            raise ValueError(
                "Missing native type or confidence outside the requested range"
            )
        source = (row.get("provenance") or {}).get("source", "unreported")
        if source not in PRODUCERS[backend] and source != "unreported":
            raise ValueError("Unexpected per-entity producer")
        sources[source] += 1
        spans.add((start, end, label))
    if len(rows) != len(spans):
        raise ValueError("Duplicate native typed span within a sentence")
    return spans, sources, runtime or "unreported"


def outcome(span, predictions):
    if span in predictions:
        return "exact"
    start, end, _ = span
    if any(a == start and b == end for a, b, _ in predictions):
        return "wrong_type"
    if any(a < end and start < b for a, b, _ in predictions):
        return "partial_overlap"
    return "no_overlap"


def first_cased_character(name):
    for char in name:
        if char.isupper():
            return "upper"
        if char.islower():
            return "lower"
    return "uncased"


def score_sentence(text, gold, predictions, names, by_arm, paired):
    def exposure(span):
        return "seen" if text[span[0] : span[1]].casefold() in names else "unseen"

    expected = {(a, b, PROJECTION[kind]) for a, b, kind in gold}
    if len(expected) != len(gold):
        raise ValueError("Projected WNUT gold spans collide")
    for role, predicted in predictions.items():
        counts = by_arm[role]
        for a, b, kind in gold:
            span = (a, b, PROJECTION[kind])
            exp = exposure(span)
            result = outcome(span, predicted)
            initial = first_cased_character(text[a:b])
            for key in ("all", exp, f"{exp}/{kind}", f"{exp}/case-{initial}"):
                counts[key]["gold"] += 1
                counts[key][result] += 1
        for span in predicted:
            for key in ("all", exposure(span)):
                counts[key]["predicted"] += 1
                counts[key]["false_positives"] += span not in expected
                counts[key]["false_positives_in_gold_empty_sentences"] += not gold
    old, new = (predictions[role] for role in ARMS)
    for label, items in (
        ("exact_gained", (new - old) & expected),
        ("exact_lost", (old - new) & expected),
        ("exact_retained", (old & new) & expected),
        ("false_positives_added", (new - old) - expected),
        ("false_positives_removed", (old - new) - expected),
    ):
        for span in items:
            for key in ("all", exposure(span)):
                paired[key][label] += 1
    if old != new:
        paired["all"]["changed_sentences"] += 1


def count_rows(counts):
    rows = {}
    for key, tally in sorted(counts.items()):
        row = {field: tally[field] for field in ("gold", *OUTCOMES)}
        if sum(row[reason] for reason in OUTCOMES) != row["gold"]:
            raise ValueError(f"Gold outcome totals disagree in {key}")
        if "/" not in key:
            for field in (
                "predicted",
                "false_positives",
                "false_positives_in_gold_empty_sentences",
            ):
                row[field] = tally[field]
            if row["exact"] + row["false_positives"] != row["predicted"]:
                raise ValueError(f"Prediction totals disagree in {key}")
        rows[key] = row
    return rows


def replay(args):
    from xxhash import xxh3_64_hexdigest

    source, cases, names = load_cases(args.data_dir)
    counts = {r: defaultdict(Counter) for r in ARMS}
    runtimes = {r: Counter() for r in ARMS}
    emitters = {r: Counter() for r in ARMS}
    predicted_types = {r: Counter() for r in ARMS}
    paired = defaultdict(Counter)
    metadata = {}
    # A private CLI configuration directory prevents a saved user profile from affecting an arm.
    with tempfile.TemporaryDirectory() as config:
        env = {
            **os.environ,
            "ANNO_NO_DOWNLOADS": "1",
            "ANNO_S3_CACHE": "0",
            "XDG_CONFIG_HOME": config,
        }
        for role in ARMS:
            binary = getattr(args, role).resolve(strict=True)
            metadata[role] = {
                "revision": getattr(args, role + "_revision"),
                "cli_version": subprocess.check_output(
                    [binary, "--version"], env=env, text=True
                ).strip(),
                "cli_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
        if metadata["baseline"]["cli_version"] != "anno 0.15.0":
            raise ValueError("The release arm must be the published anno-cli 0.15.0")
        for index, (text, gold) in enumerate(cases):
            predictions = {}
            text_hash = "xxh3:" + xxh3_64_hexdigest(text.encode("utf-8"))
            for role in ARMS:
                cmd = [
                    getattr(args, role).resolve(),
                    "extract",
                    "--quiet",
                    *(["--model", "heuristic"] if args.backend == "heuristic" else []),
                    "--format",
                    "json",
                    "--threshold",
                    "0.5",
                    "--text=" + text,
                ]
                result = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
                if result.returncode:
                    raise RuntimeError(
                        f"{role} exited {result.returncode} on WNUT test sentence {index}"
                    )
                prediction, sources, runtime = check_prediction(
                    json.loads(result.stdout),
                    text,
                    text_hash,
                    role == "candidate",
                    args.backend,
                )
                predictions[role] = prediction
                emitters[role].update(sources)
                if args.backend == "stacked":
                    predicted_types[role].update(label for _, _, label in prediction)
                runtimes[role][runtime] += 1
            score_sentence(text, gold, predictions, names, counts, paired)
    for role in ARMS:
        if not PRODUCERS[args.backend].issubset(emitters[role]):
            raise ValueError(f"Missing expected {args.backend} producers in {role}")
        metadata[role].update(
            runtime_model_calls=dict(runtimes[role]),
            entity_sources=dict(emitters[role]),
            counts=count_rows(counts[role]),
        )
        if args.backend == "stacked":
            metadata[role]["predictions_by_type"] = dict(predicted_types[role])
    changes = {
        k: {
            field: paired[k][field]
            for field in (
                "exact_gained",
                "exact_lost",
                "exact_retained",
                "false_positives_added",
                "false_positives_removed",
            )
        }
        for k in ("all", "seen", "unseen")
    }
    changes["all"]["changed_sentences"] = paired["all"]["changed_sentences"]
    return {
        "source": source,
        "protocol": {
            "backend": args.backend,
            **(
                {"model_selection": "CLI default (omitted --model)"}
                if args.backend == "stacked"
                else {}
            ),
            "threshold": 0.5,
            "cargo_default_features": False,
            "cargo_features": ["extractor-html2text"],
            "gold_projection": PROJECTION,
            "offset_unit": "Unicode code points, half-open",
            "token_separator": "U+0020",
            "exposure": "casefolded complete surface present among training gold mentions",
        },
        "arms": metadata,
        "paired": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ARMS:
        parser.add_argument(
            "--" + role, type=Path, required=True, help=f"Native {role} anno executable"
        )
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--candidate-revision", default=CANDIDATE_REVISION)
    parser.add_argument(
        "--backend",
        choices=tuple(RUNTIME),
        default="heuristic",
        help="For stacked, omit --model and verify the effective CLI default",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Pinned WNUT checkout. Fetch publisher files if omitted.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
