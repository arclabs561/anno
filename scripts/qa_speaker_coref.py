#!/usr/bin/env python3
"""Replay first-person mentions from two AMI meetings and score the labeled pairs.

Run python3 scripts/qa_speaker_coref.py --anno target/debug/anno.
Use --text to print the exact rendered CLI input, or --predictions FILE to
score saved JSON from `anno pipeline --coref --format json`. Use --speaker-turns
with a CLI that supports explicit annotations. Exit 1 means
one or more labeled pairs failed or a mention could not be resolved.
"""

import argparse
import json
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[1] / "crates/anno-cli/tests/fixtures"


def transcript(meeting, label):
    fixture = FIXTURES / f"ami_{meeting}_coref.json"
    turns = json.loads(fixture.read_text(encoding="utf-8"))["turns"]
    lines = []
    sidecar = []
    gold = []
    offset = 0
    for turn in turns:
        prefix = turn[label] + ": "
        line = prefix + turn["excerpt"]
        start, end = turn["span"]
        if turn["excerpt"][start:end] != turn["mention"]:
            raise ValueError(f"Bad gold span in {turn['segment']}")
        body_start = offset + len(prefix)
        sidecar.append({"speaker": turn[label], "text": turn["excerpt"],
                        "start": body_start, "end": offset + len(line)})
        gold.append((turn["channel"], (body_start + start, body_start + end)))
        lines.append(line)
        offset += len(line) + 1
    return "\n".join(lines), gold, sidecar


def score(doc, text, gold):
    if doc["text"] != text:
        raise ValueError("Prediction text differs from the selected fixture rendering")
    signals = {signal["id"]: signal for signal in doc["signals"]}
    memberships = {}
    for track_id, track in doc["tracks"].items():
        for ref in track["signals"]:
            signal = signals[ref["signal_id"]]
            location = signal["location"].get("Text")
            if location:
                span = location["start"], location["end"]
                if text[slice(*span)] != signal["surface"]:
                    raise ValueError(f"Signal surface disagrees with source span {span}")
                memberships.setdefault(span, set()).add(track_id)

    # A missing or multiply assigned mention supplies no evidence about either
    # pair type. In particular, empty predictions cannot satisfy cannot-links.
    tracks = []
    for _, span in gold:
        ids = memberships.get(span, set())
        tracks.append(next(iter(ids)) if len(ids) == 1 else None)
    totals = {relation: dict(correct=0, wrong=0, unresolved=0)
              for relation in ("same_speaker", "different_speaker")}
    for i, j in combinations(range(len(gold)), 2):
        same = gold[i][0] == gold[j][0]
        counts = totals["same_speaker" if same else "different_speaker"]
        if tracks[i] is None or tracks[j] is None:
            counts["unresolved"] += 1
        elif (tracks[i] == tracks[j]) == same:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return {
        "coref_backend": doc.get("coref_backend"),
        "gold_mentions": len(gold),
        "unambiguously_tracked": sum(track is not None for track in tracks),
        **totals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--turns", action="store_true", help="write the speaker-turn JSON sidecar")
    mode.add_argument("--text", action="store_true", help="write the exact CLI input")
    mode.add_argument("--predictions", type=Path, help="score a saved one-document pipeline JSON array")
    mode.add_argument("--anno", default="anno", help="path to the anno executable")
    parser.add_argument("--label", choices=["speaker", "channel"], default="speaker")
    parser.add_argument("--meeting", choices=["es2002a", "es2003a"], default="es2002a",
                        help="es2003a was held out while developing the structured resolver")
    parser.add_argument("--speaker-turns", action="store_true",
                        help="supply the gold speaker-turn sidecar to the CLI")
    args = parser.parse_args()
    if args.speaker_turns and (args.text or args.turns or args.predictions):
        parser.error("--speaker-turns applies only when invoking the CLI")
    text, gold, sidecar = transcript(args.meeting, args.label)
    if args.turns:
        print(json.dumps(sidecar, ensure_ascii=False, indent=2))
        return 0
    if args.text:
        print(text, end="")
        return 0
    if args.predictions:
        documents = json.loads(args.predictions.read_text(encoding="utf-8"))
    else:
        command = [args.anno, "pipeline", "--model", "heuristic", "--coref", "--format", "json", "--quiet"]
        with tempfile.TemporaryDirectory() as directory:
            if args.speaker_turns:
                source = Path(directory) / "transcript.txt"
                turns = Path(directory) / "turns.json"
                source.write_text(text, encoding="utf-8")
                turns.write_text(json.dumps(sidecar), encoding="utf-8")
                command.extend(["--speaker-turns", str(turns), "--files", str(source)])
            else:
                command.append(text)
            process = subprocess.run(command, check=True, stdout=subprocess.PIPE)
        documents = json.loads(process.stdout)
    [doc] = documents
    report = score(doc, text, gold)
    print(json.dumps({"meeting": args.meeting, **report}, indent=2))
    return int(any(report[key]["wrong"] or report[key]["unresolved"]
                   for key in ("same_speaker", "different_speaker")))


if __name__ == "__main__":
    raise SystemExit(main())
