"""Compare the two-column NorNE cache and the publisher's CoNLL-U in native Anno."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from english_ner_source_replay import canonical, metric, native_gold_key, predict, sha

ROOT = Path(__file__).resolve().parents[2]
MODEL = "heuristic"
SOURCE_REVISION = "0c0d133a0f9590d76b81078712261badd4b605e9"
BASELINE_REVISION = "136a7e7aea5c8abc6b722822c2b5f5473f5ede88"
TYPES = {"PER", "LOC", "ORG", "GPE_LOC", "GPE_ORG", "EVT", "PROD", "MISC", "DRV"}
SEVEN_TYPE = {"GPE_LOC": "LOC", "GPE_ORG": "ORG"}
Span = tuple[int, int, str]


@dataclass
class Sentence:
    identity: str
    source: str
    joined: str
    tokens: list[tuple[str, str]]
    source_gold: list[Span]
    joined_gold: list[Span]
    added_spaces: list[int]

    def on_source(self, span):
        start, end, label = span
        return (
            start - bisect_left(self.added_spaces, start),
            end - bisect_left(self.added_spaces, end),
            label,
        )


def sentence(block):
    """Read literal name tags and locate each form in the publisher's sentence header."""
    headers, words = {}, []
    for line in block.split("\n"):
        if line.startswith("#"):
            key, sep, value = line.partition(" = ")
            if sep and key in {"# text", "# sent_id"}:
                if key in headers:
                    raise ValueError("Repeated NorNE sentence header")
                headers[key] = value
            continue
        fields = line.split("\t")
        if len(fields) != 10 or fields[0] != str(len(words) + 1):
            raise ValueError(
                "Expected consecutively numbered word rows in pinned NorNE"
            )
        attrs = fields[-1].split("|")
        names = [attr[5:] for attr in attrs if attr.startswith("name=")]
        if len(names) != 1 or not fields[1]:
            raise ValueError("Expected a nonempty form and exactly one name attribute")
        words.append((fields[1], names[0], "SpaceAfter=No" in attrs))
    if not words or headers.keys() != {"# text", "# sent_id"}:
        raise ValueError("Missing NorNE words, sentence identifier or original text")
    source = headers["# text"]
    added_spaces, tokens, source_gold, joined_gold = [], [], [], []
    position, previous = 0, "O"
    for i, (word, tag, no_space) in enumerate(words):
        if not source.startswith(word, position):
            raise ValueError("Word forms or declared spacing disagree with # text")
        end = position + len(word)
        old_start, old_end = position + len(added_spaces), end + len(added_spaces)
        if tag != "O":
            prefix, sep, label = tag.partition("-")
            if not sep or label not in TYPES or prefix not in {"B", "I"}:
                raise ValueError("Unknown literal NorNE entity tag")
            if prefix == "B":
                source_gold.append((position, end, label))
                joined_gold.append((old_start, old_end, label))
            else:
                if previous not in {"B-" + label, tag}:
                    raise ValueError("Orphan NorNE inside tag")
                source_gold[-1] = (source_gold[-1][0], end, label)
                joined_gold[-1] = (joined_gold[-1][0], old_end, label)
        previous = tag
        tokens.append((word, tag))
        position = end
        if i + 1 < len(words):
            if no_space:
                added_spaces.append(old_end)
            elif source[end : end + 1] == " ":
                position += 1
            else:
                raise ValueError("Expected a space between publisher tokens")
    if position != len(source):
        raise ValueError("Unaccounted characters in original NorNE sentence")
    joined = " ".join(t for t, _ in tokens)
    return Sentence(
        headers["# sent_id"],
        source,
        joined,
        tokens,
        source_gold,
        joined_gold,
        added_spaces,
    )


def legacy_bytes(cases):
    return "".join(
        "".join(f"{word}\t{tag}\n" for word, tag in case.tokens) + "\n"
        for case in cases
    ).encode()


@dataclass
class Observation:
    typed: list[Span]
    exact: set[int]


def evaluate(binary, cases, payload, original, work):
    """Check native summaries and actual predictions against the source-bound reference."""
    env = {
        **os.environ,
        "ANNO_CACHE_DIR": str(work),
        "ANNO_NO_DOWNLOADS": "1",
        "ANNO_S3_CACHE": "0",
        "NO_COLOR": "1",
    }
    (work / "datasets" / "NorNE.cache").write_bytes(payload)
    reports = work / "reports.jsonl"
    subprocess.run(
        [
            binary,
            "dataset",
            "eval",
            "--cached-only",
            "--dataset",
            "NorNE",
            "--task",
            "ner",
            "--model",
            MODEL,
            "--reports",
            reports,
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    batch = predict(
        binary,
        MODEL,
        [(c.identity, c.source if original else c.joined, None) for c in cases],
        env,
    )
    measured = []
    with reports.open(encoding="utf-8") as stream:
        for index, (case, prediction, row) in enumerate(
            zip(cases, batch, stream, strict=True), 1
        ):
            report = json.loads(row)
            gold = case.source_gold if original else case.joined_gold
            text = case.source if original else case.joined
            digest = sha(text.encode())
            keys = {
                str(i): native_gold_key(label, a, b)
                for i, (a, b, label) in enumerate(gold)
            }
            pair = report["pairing"]
            if (
                pair["source_sha256"],
                pair["document_id"],
                pair["gold_keys"],
                report["model"],
            ) != (digest, f"NorNE:{index}", keys, MODEL):
                raise ValueError(
                    f"NorNE:{index}: native input, entity coordinates or labels differ"
                )
            if (
                prediction["source_sha256"],
                prediction["id"],
                prediction["text_length"],
                prediction["model"],
                prediction["runtime_model"],
            ) != (digest, case.identity, len(text), MODEL, MODEL):
                raise ValueError(
                    f"NorNE:{index}: batch input or observed producer differs"
                )
            typed = []
            for entity in prediction["entities"]:
                a, b = entity["start"], entity["end"]
                if not 0 <= a < b <= len(text) or text[a:b] != entity["text"]:
                    raise ValueError(f"NorNE:{index}: invalid native Unicode extent")
                typed.append((a, b, entity["type"]))
            observed = set(typed)
            exact = {i for i, span in enumerate(gold) if span in observed}
            matched = {
                m["gold_id"] for m in report["matches"] if m["type"] == "correct"
            }
            if exact != matched or (len(exact), len(gold), len(typed), len(typed)) != (
                report["correct"],
                report["gold_count"],
                report["predicted_count"],
                prediction["entity_count"],
            ):
                raise ValueError(
                    f"NorNE:{index}: native counts or exact mention identities differ"
                )
            measured.append(Observation(typed, exact))
    return measured


def compare_predictions(case, before, after):
    """Pair typed suggestions by source span, with multiplicity; retain unchanged offsets first."""
    same, shifted = Counter(), Counter()
    for span in before:
        projected = case.on_source(span)
        (same if span == projected else shifted)[projected] += 1
    reference = Counter(after)
    same_hits = (same & reference).total()
    shifted_hits = (shifted & (reference - same)).total()
    paired = same_hits + shifted_hits
    return Counter(
        {
            "retained_same_offsets": same_hits,
            "retained_shifted_offsets": shifted_hits,
            "added": len(after) - paired,
            "removed": len(before) - paired,
        }
    )


def miss_kind(text, gold, predicted):
    """Classify one missed exact gold without counting competing predictions twice."""
    start, end, label = gold
    overlaps = [(a, b, t) for a, b, t in predicted if a < end and start < b]
    if any((a, b) == (start, end) for a, b, _ in overlaps):
        return "wrong_type_at_exact_boundary"
    for a, b, t in overlaps:
        if t == label and a <= start and b >= end:
            outside = text[a:start] + text[end:b]
            if outside and all(
                unicodedata.category(c).startswith("P") for c in outside
            ):
                return "same_type_extended_by_punctuation_only"
    return "other_overlap" if overlaps else "no_overlap"


def summarize(cases, before, after):
    gold_pairs = Counter(retained=0, gained=0, lost=0)
    coordinates = Counter()
    paired_predictions, by_gold, by_hit, by_bounds, by_prediction = (
        Counter() for _ in range(5)
    )
    same_boundary_types, missed_by_type = defaultdict(Counter), defaultdict(Counter)
    paired_loss_kinds = Counter()
    seven_type_correct = 0
    ordered_sources = hashlib.sha256()
    ordered_predictions = {arm: hashlib.sha256() for arm in ("before", "after")}
    for case, old, new in zip(cases, before, after, strict=True):
        if case.source == case.joined and old != new:
            raise ValueError(
                "Unchanged input produced different heuristic suggestions or judgments"
            )
        if any(
            case.on_source(s) != t
            for s, t in zip(case.joined_gold, case.source_gold, strict=True)
        ):
            raise ValueError(
                "Source-token gold identities do not align across coordinate systems"
            )
        coordinates.update(
            cases=1,
            tokens=len(case.tokens),
            gold=len(case.source_gold),
            source_text_exact_before=int(case.source == case.joined),
            source_text_exact_after=1,
            source_gold_exact_before=sum(
                s == t for s, t in zip(case.joined_gold, case.source_gold)
            ),
            source_gold_exact_after=len(case.source_gold),
            artificial_spaces=len(case.added_spaces),
        )
        gold_pairs.update(
            retained=len(old.exact & new.exact),
            lost=len(old.exact - new.exact),
            gained=len(new.exact - old.exact),
        )
        predictions = compare_predictions(case, old.typed, new.typed)
        paired_predictions.update(predictions)
        coordinates["cases_with_changed_suggestions"] += int(
            bool(predictions["added"] or predictions["removed"])
        )
        by_prediction.update(t for _, _, t in new.typed)
        at_boundary = defaultdict(set)
        for a, b, label in new.typed:
            at_boundary[a, b].add(label)
        seven_type_predictions = {(a, b, SEVEN_TYPE.get(t, t)) for a, b, t in new.typed}
        for i, (a, b, label) in enumerate(case.source_gold):
            seven_type_correct += (
                a,
                b,
                SEVEN_TYPE.get(label, label),
            ) in seven_type_predictions
            by_gold[label] += 1
            by_hit[label] += i in new.exact
            by_bounds[label] += bool(at_boundary[a, b])
            same_boundary_types[label].update(at_boundary[a, b])
            if i not in new.exact:
                reason = miss_kind(case.source, (a, b, label), new.typed)
                missed_by_type[label][reason] += 1
                if i in old.exact:
                    paired_loss_kinds[reason] += 1
        ordered_sources.update(
            canonical([case.identity, sha(case.source.encode())]) + b"\n"
        )
        for arm, record in [("before", old), ("after", new)]:
            ordered_predictions[arm].update(
                canonical([case.identity, record.typed]) + b"\n"
            )
    result = {
        **coordinates,
        "ordered_source_sha256": ordered_sources.hexdigest(),
        "paired_gold": dict(gold_pairs),
        "paired_gold_loss_on_source": dict(paired_loss_kinds),
        "paired_predictions": dict(paired_predictions),
        "after_gold_by_type": {
            label: {
                "gold": by_gold[label],
                "correct": by_hit[label],
                "exact_boundary_any_type": by_bounds[label],
                "types_at_exact_boundary": dict(same_boundary_types[label]),
                "missed": dict(missed_by_type[label]),
            }
            for label in sorted(by_gold)
        },
        "after_predictions_by_type": dict(by_prediction),
    }
    for arm, observed in [("before", before), ("after", after)]:
        result[arm] = metric(
            coordinates["gold"],
            sum(len(o.typed) for o in observed),
            sum(len(o.exact) for o in observed),
        )
        result[arm]["ordered_predictions_sha256"] = ordered_predictions[arm].hexdigest()
    result["after_seven_type"] = metric(
        coordinates["gold"], result["after"]["predicted"], seven_type_correct
    )
    return result


def native_test_errors(binary, path, splits):
    raw = subprocess.check_output(
        [
            binary,
            "viz",
            "errors",
            path,
            "--format",
            "json",
            "--title",
            "Which Norwegian place names does the heuristic mistype?",
        ]
    )
    summary = json.loads(raw)
    tests = [splits["nob-test"], splits["nno-test"]]
    stats = summary["entities"]
    if summary["reports"] != sum(t["cases"] for t in tests) or any(
        stats[field] != sum(t["after"][key] for t in tests)
        for field, key in (
            ("exact", "correct"),
            ("gold_count", "gold"),
            ("predicted_count", "predicted"),
        )
    ):
        raise ValueError(
            "Portable error chart and independently paired test counts disagree"
        )
    expected = Counter()
    for panel in tests:
        for label, by_type in panel["after_gold_by_type"].items():
            for predicted, count in by_type["types_at_exact_boundary"].items():
                if label != predicted:
                    expected[label, predicted] += count
    observed = {
        (r["gold"], r["predicted"]): r["pairs"]
        for r in summary["same_span_label_confusions"]
    }
    if observed != expected:
        raise ValueError(
            "Portable chart reports different source-to-prediction type confusions"
        )
    return raw


def study(args):
    registry = json.loads(
        (ROOT / "scripts/data/offline-native-sources.json").read_bytes()
    )
    spec = next(d for d in registry["datasets"] if d["id"] == "NorNE")
    expected = {
        f"{lang}-{split}"
        for lang in ("nno", "nob")
        for split in ("train", "dev", "test")
    }
    if sorted(item["variant"] for item in spec["inputs"]) != sorted(expected):
        raise ValueError(
            "Expected all six NorNE split and written standard combinations"
        )
    splits = {}
    with zipfile.ZipFile(args.archive) as archive, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "datasets").mkdir()
        profiles = {
            r["variant"]: r
            for r in json.loads(archive.read("sources.json"))["inputs"]
            if r["id"] == "NorNE"
        }
        if set(profiles) != expected:
            raise ValueError(
                "Supply a native archive with all six original NorNE release files"
            )
        test_reports = work / "test-reports.jsonl"
        for item in spec["inputs"]:
            variant, url = item["variant"], item["url"]
            profile = profiles[variant]
            raw = archive.read(profile["original_path"])
            if SOURCE_REVISION not in url.split("/") or sha(raw) != item["sha256"]:
                raise ValueError(
                    f"Unexpected publisher file revision or SHA-256 for {variant}"
                )
            cases = [
                sentence(block) for block in raw.decode().rstrip("\n").split("\n\n")
            ]
            if len({case.identity for case in cases}) != len(cases):
                raise ValueError(f"Duplicate publisher sentence ID in {variant}")
            old_payload = legacy_bytes(cases)
            if profile.get("projection") == "norne-bio" and old_payload != archive.read(
                profile["path"]
            ):
                raise ValueError(
                    f"Legacy BIO reconstruction differs from archived cache on {variant}"
                )
            before = evaluate(args.anno, cases, old_payload, False, work)
            if args.prior_anno and before != evaluate(
                args.prior_anno, cases, old_payload, False, work
            ):
                raise ValueError(
                    f"Legacy binary predictions or exact identities differ on {variant}"
                )
            after = evaluate(args.anno, cases, raw, True, work)
            if args.errors_output and item["split"] == "test":
                with (
                    (work / "reports.jsonl").open("rb") as reports,
                    test_reports.open("ab") as pooled,
                ):
                    shutil.copyfileobj(reports, pooled)
            splits[variant] = {
                "source_url": url,
                "source_sha256": sha(raw),
                "legacy_bio_sha256": sha(old_payload),
                **summarize(cases, before, after),
            }
        if args.errors_output:
            args.errors_output.write_bytes(
                native_test_errors(args.anno, test_reports, splits)
            )
    return {
        "schema": "anno.norne-source-replay/v1",
        "dataset": "NorNE",
        "backend": MODEL,
        "anno_version": subprocess.check_output(
            [args.anno, "--version"], text=True
        ).strip(),
        "source_revision": SOURCE_REVISION,
        "baseline_revision": BASELINE_REVISION,
        "scoring": "Literal name types and exact Unicode character boundaries on each arm's text. Pair gold by source split, sentence and mention ordinal. Pair typed predictions after removing inserted inter-token spaces from the legacy coordinates.",
        "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Complete native NorNE ZIP with all six original CoNLL-U files",
    )
    parser.add_argument(
        "--anno",
        type=Path,
        required=True,
        help="Eval-enabled Anno binary supporting original NorNE",
    )
    parser.add_argument(
        "--prior-anno",
        type=Path,
        help="Also verify the legacy arm with an earlier eval-enabled binary",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--errors-output",
        type=Path,
        help="Write a portable anno.errors/v1 chart for the two complete publisher test splits",
    )
    args = parser.parse_args()
    args.anno = args.anno.resolve()
    if args.prior_anno:
        args.prior_anno = args.prior_anno.resolve()
    args.output.write_text(
        json.dumps(study(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
