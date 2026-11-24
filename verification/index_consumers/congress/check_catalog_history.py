# SPDX-License-Identifier: MIT OR Apache-2.0
"""Run the historical correction through separate anno processes; optionally time Clerk inputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
import tempfile
import time

from publication_input import grounded_names


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def build_cli():
    command = ["cargo", "build", "--locked", "-p", "anno-cli", "--no-default-features",
               "--features", "extractor-readability",
               "--bin", "anno", "--message-format=json"]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, check=True, text=True)
    for line in result.stdout.splitlines():
        artifact = json.loads(line)
        if artifact.get("target", {}).get("name") == "anno" and artifact.get("executable"):
            return Path(artifact["executable"])
    raise RuntimeError("Cargo did not report the path to the anno binary")


def query(binary, current, *, prior=None, record=None):
    command = [str(binary), "query", "publication", str(current)]
    if prior is not None:
        command += ["--since", str(prior)]
    if record is not None:
        command += ["--catalog-record", *record]
    response = subprocess.run(command, text=True, capture_output=True)
    if response.returncode:
        raise RuntimeError(response.stderr.strip())
    return json.loads(response.stdout)


def check_historical_correction(binary, directory):
    fixture = json.loads((HERE / "attribution-correction.json").read_text())
    source = fixture["source"]
    catalog = fixture["catalog"]
    namespace = catalog["namespace"]
    text = source["text"]
    assert hashlib.sha256(text.encode()).hexdigest() == source["source_text_sha256"]
    assert text[source["start"]:source["end"]] == source["quote"]

    def surface_matches(revision):
        return [key for key, name in catalog[revision]["names"].items()
                if f"{name['first']} {name['last']}" == source["quote"]]

    original, = surface_matches("before")
    assert original == fixture["prior_binding"]["accession"]
    assert not surface_matches("after")
    successor = fixture["surviving_binding"]["evidence"]["value"]
    assert successor in catalog["after"]["names"] and original not in catalog["after"]["names"]

    for case in fixture["cases"]:
        stream = f"historical-catalog/{case['id']}"

        def write(name, sequence, accession):
            path = directory / name
            doc = grounded_names(source["id"], [(text, accession)], catalog=namespace)
            path.write_text(json.dumps({"stream": stream, "sequence": sequence,
                                        "source_revision": str(source["wikidata_revision"]), "document": doc}))
            return path

        before = write("before.json", 1, original)
        accepted = successor if case["id"] == "explicit_scoped_rebind" else None
        after = write("after.json", 2, accepted)
        old_key = (namespace, original)
        new_key = (namespace, successor)
        initial = query(binary, before, record=old_key)
        assert len(initial["rows"]) == 1
        # A reader keyed only by the unchanged source digest still shows this wrong subject.
        assert initial["rows"][0]["source_sha256"] == source["source_text_sha256"]
        current = query(binary, after)
        row, = current["rows"]
        assert row["source_revision"] == str(source["wikidata_revision"])
        assert row["surface"] == source["quote"] and row["source_sha256"] == source["source_text_sha256"]
        assert row["location"]["Text"] == {"start": source["start"], "end": source["end"]}
        assert row["catalog_record"] == ({"catalog": namespace, "accession": accepted} if accepted else None)
        assert len(query(binary, after, record=old_key)["rows"]) == case["expected_old_hits"]
        assert len(query(binary, after, record=new_key)["rows"]) == case["expected_surviving_hits"]
        removed = query(binary, after, prior=before, record=old_key)
        assert removed["base"] == initial["checkpoint"]
        assert removed["removed"] == initial["rows"] and not removed["upserted"]
        added = query(binary, after, prior=before, record=new_key)
        assert len(added["upserted"]) == case["expected_surviving_hits"] and not added["removed"]
        replay = query(binary, after, prior=after)
        assert replay["base"] == replay["target"] == current["checkpoint"]
        assert not replay["removed"] and not replay["upserted"]
        # The same sequence diverges in the old record's view. A subscriber to an
        # unrelated record has identical projected data and should observe no fork.
        fork = write("same-sequence.json", 1, accepted)
        hidden = query(binary, fork, prior=before, record=("another-catalog", original))
        assert hidden["base"] == hidden["target"] and not hidden["removed"] and not hidden["upserted"]
        try:
            query(binary, fork, prior=before, record=old_key)
        except RuntimeError as error:
            assert "conflicting contents" in str(error)
        else:
            raise AssertionError("accepted a conflicting revision of a visible posting")
        print(f"{case['id']}: verified unchanged quote and published catalog rows")


def publish(writer, receipt):
    """Install one CLI receipt and its checkpoint in this single-stream SQLite reader."""
    with writer:
        writer.execute("BEGIN IMMEDIATE")
        current = writer.execute("SELECT checkpoint FROM head").fetchone()
        if "rows" in receipt:
            assert current is None
            target, additions = receipt["checkpoint"], receipt["rows"]
        else:
            target, additions = receipt["target"], receipt["upserted"]
            if json.loads(current[0]) == target:
                return
            assert json.loads(current[0]) == receipt["base"]
            writer.executemany("DELETE FROM postings WHERE signal_id = ?",
                               ((row["signal_id"],) for row in receipt["removed"]))
        writer.executemany(
            "INSERT INTO postings VALUES (?, ?, ?, ?)",
            ((row["signal_id"], row["surface"],
              (row["catalog_record"] or {}).get("catalog"),
              (row["catalog_record"] or {}).get("accession")) for row in additions),
        )
        writer.execute("DELETE FROM head")
        writer.execute("INSERT INTO head VALUES (?)", (json.dumps(target),))


def time_roll_call(binary, directory, repeats):
    before = directory / "house-2023-001-before.json"
    after = directory / "house-2023-001-after.json"
    durations = {"first name lookup": [], "corrected catalog lookup": []}
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "roll.sqlite"
            writer, reader = sqlite3.connect(db), sqlite3.connect(db)
            writer.executescript("""
                CREATE TABLE head (checkpoint TEXT);
                CREATE TABLE postings (signal_id INTEGER PRIMARY KEY, name TEXT, catalog TEXT, accession TEXT);
                CREATE INDEX by_name ON postings (name);
                CREATE INDEX by_subject ON postings (catalog, accession);
            """)
            start = time.perf_counter_ns()
            initial = query(binary, before)
            publish(writer, initial)
            literal = reader.execute("SELECT accession FROM postings WHERE name = ?", ("Garcia, Robert",)).fetchall()
            durations["first name lookup"].append((time.perf_counter_ns() - start) / 1e6)
            assert len(initial["rows"]) == 435 and literal == [(None,)]
            assert all(row["catalog_record"] is None for row in initial["rows"])

            start = time.perf_counter_ns()
            delta = query(binary, after, prior=before)
            publish(writer, delta)
            attributed = reader.execute("SELECT name FROM postings WHERE catalog = ? AND accession = ?",
                                        ("bioguide", "G000598")).fetchall()
            durations["corrected catalog lookup"].append((time.perf_counter_ns() - start) / 1e6)
            assert attributed == [("Garcia, Robert",)]
            assert len(delta["removed"]) == len(delta["upserted"]) == 430
            assert all(row["catalog_record"] is not None for row in delta["upserted"])
            assert reader.execute("SELECT COUNT(*), COUNT(accession) FROM postings").fetchone() == (435, 430)
            writes = writer.total_changes
            publish(writer, delta)
            assert writer.total_changes == writes
            writer.close()
            reader.close()
    for label, samples in durations.items():
        p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
        print(f"{label} ({repeats} trials through CLI, SQLite commit and reader): "
              f"median={statistics.median(samples):.2f} ms, p95={p95:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, help="prebuilt anno binary (default: build locally)")
    parser.add_argument("--roll-calls", type=Path, help="optional envelopes from check_rollcalls.py")
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    binary = args.anno.resolve() if args.anno else build_cli()
    with tempfile.TemporaryDirectory() as directory:
        check_historical_correction(binary, Path(directory))
    if args.roll_calls:
        time_roll_call(binary, args.roll_calls, args.repeats)


if __name__ == "__main__":
    main()
