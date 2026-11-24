# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exercise coherent editions through fresh CLI processes, with independent bulk reads."""
import argparse
import copy
import itertools
import json
import multiprocessing
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "congress"))
from check_catalog_history import build_cli, query
from publication_input import grounded_names

NAMES = ("Arden", "Blair", "Cato")
SEQUENCES = ((9, 11, 17), (31, 50, 98), (2, 8, 64))


def save(path, data):
    path.write_text(json.dumps(data))
    return path


class Client:
    def __init__(self, binary, audience="public"):
        self.command = [str(binary), "query", "edition", "--collection", "program",
                        "--source", "editor", "--audience", audience]

    def call(self, *args, error=None):
        result = subprocess.run([*self.command, *map(str, args)], capture_output=True, text=True)
        if error is not None:
            assert result.returncode and error in result.stderr, (args, result)
            return
        assert result.returncode == 0, (args, result.stderr)
        return json.loads(result.stdout)


def prepare(binary, temp, client):
    snapshots, deltas, manifests, bulk = {}, {}, {}, {}
    for person, name in enumerate(NAMES):
        prev = None
        for edition, seq in enumerate(SEQUENCES[person]):
            key = person, edition
            doc = grounded_names(name, [(name, "lead" if edition == person else None)], catalog="credit")
            path = save(temp / f"doc-{person}-{edition}", dict(stream="program", sequence=seq,
                        source_revision="printed-1", document=doc))
            snapshot = query(binary, path, record=("credit", "lead"))
            snapshots[key] = save(temp / f"snapshot-{person}-{edition}", snapshot)
            if prev:
                deltas[key] = save(temp / f"delta-{person}-{edition}",
                                   query(binary, path, prior=prev, record=("credit", "lead")))
            prev = path
    for edition in range(3):
        paths = [snapshots[person, edition] for person in range(3)]
        bulk[edition] = sorted((json.loads(p.read_text()) for p in paths),
                               key=lambda s: s["checkpoint"]["stream"])
        assert [r["surface"] for m in bulk[edition] for r in m["rows"]] == [NAMES[edition]]
        manifests[edition] = save(temp / f"manifest-{edition}", client.call("manifest", edition, *paths))
    # The endpoint delta for Blair omits the intermediate published lead credit.
    net = query(binary, temp/"doc-1-2", prior=temp/"doc-1-0", record=("credit", "lead"))
    assert net["removed"] == net["upserted"] == []
    assert json.loads(deltas[1, 1].read_text())["upserted"][0]["surface"] == "Blair"
    assert json.loads(deltas[1, 2].read_text())["removed"][0]["surface"] == "Blair"
    return snapshots, deltas, manifests, bulk


def schedules(client, temp, snapshots, deltas, manifests, bulk):
    base = temp / "baseline.sqlite"
    for p in range(3):
        client.call("stage", base, snapshots[p, 0])
    client.call("publish", base, manifests[0])
    reads = 0
    for order in sorted(set(itertools.permutations((0, 0, 1, 1, 2, 2)))):
        db = temp / "reader.sqlite"
        shutil.copyfile(base, db)
        positions = [0, 0, 0]
        published = 0
        for p in order:
            positions[p] += 1
            client.call("stage", db, deltas[p, positions[p]])
            attempted = client.call("publish", db, manifests[published + 1])
            expected = min(positions)
            assert attempted["status"] == ("waiting" if expected == published else "published")
            published = expected
            view = client.call("read", db)
            assert view["manifest"] == json.loads(manifests[expected].read_text())
            assert view["members"] == bulk[expected], (order, positions, view)
            reads += 1
    assert reads == 540
    return base


def observe_pending(db):
    # SQLite coalesces SHARED locks between connections in one process. Use an
    # independent process to prove that the writer is at its final commit lock.
    deadline = time.monotonic() + 3
    with sqlite3.connect(db, timeout=0) as observer:
        while time.monotonic() < deadline:
            try:
                observer.execute("SELECT count(*) FROM published_editions").fetchone()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc):
                    raise
                return
            time.sleep(0.002)
    raise AssertionError("writer did not reach its pending commit")


def crash_before_commit(client, db, manifest):
    # A SHARED read transaction lets a writer prepare its changes and blocks its
    # final EXCLUSIVE lock. A PENDING commit then refuses new SHARED readers.
    with sqlite3.connect(db, isolation_level=None) as reader:
        reader.execute("BEGIN")
        old = reader.execute("SELECT count(*) FROM published_editions").fetchone()
        writer = subprocess.Popen([*client.command, "publish", str(db), str(manifest)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            probe = multiprocessing.get_context("spawn").Process(target=observe_pending, args=(db,))
            probe.start()
            probe.join(timeout=5)
            assert probe.exitcode == 0, probe.exitcode
            assert writer.poll() is None, writer.communicate()
            assert reader.execute("SELECT count(*) FROM published_editions").fetchone() == old
            writer.kill()
            writer.communicate(timeout=5)
        finally:
            if writer.poll() is None:
                writer.kill()
                writer.communicate()
            reader.rollback()


def failure_cases(binary, client, temp, baseline, snapshots, deltas, manifests, bulk):
    db = temp / "recovery.sqlite"
    shutil.copyfile(baseline, db)
    for p in (0, 1):
        client.call("stage", db, deltas[p, 1])
    missing = client.call("publish", db, manifests[1])
    withheld = json.loads(snapshots[2, 1].read_text())
    assert withheld["rows"] == [] and missing["missing"] == [withheld["checkpoint"]]
    assert client.call("read", db)["members"] == bulk[0]
    corrupt = copy.deepcopy(withheld)
    corrupt["source_revision"] = "altered"
    client.call("stage", db, save(temp/"bad-empty", corrupt), error="native checkpoint")
    corrupt = json.loads(deltas[2, 2].read_text())
    corrupt["upserted"][0]["surface"] = "altered"
    client.call("stage", db, deltas[2, 1])
    client.call("stage", db, save(temp/"bad-delta", corrupt), error="native checkpoint")
    client.call("stage", db, deltas[2, 1])
    crash_before_commit(client, db, manifests[1])
    assert client.call("read", db)["members"] == bulk[0]
    assert client.call("publish", db, manifests[1])["status"] == "published"
    assert client.call("publish", db, manifests[1])["status"] == "already_published"
    assert client.call("read", db, 1)["members"] == bulk[1]

    # An accepted stream sequence cannot be rebound, even by a genuine native snapshot.
    fork = json.loads((temp/"doc-0-1").read_text())
    fork["sequence"] = SEQUENCES[0][0]
    path = save(temp/"fork-doc", fork)
    snapshot = query(binary, path, record=("credit", "lead"))
    client.call("stage", db, save(temp/"fork-receipt", snapshot), error="different checkpoint")

    old = json.loads(snapshots[0, 0].read_text())
    old_cp = save(temp/"old-checkpoint", old["checkpoint"])
    assert client.call("evict", db, old_cp)["evicted"]
    client.call("read", db, 0, error="evicted member")
    client.call("stage", db, deltas[0, 1], error="base unavailable")
    client.call("stage", db, snapshots[0, 0])
    assert client.call("read", db, 0)["members"] == bulk[0]
    client.call("evict", db, save(temp/"active-checkpoint", withheld["checkpoint"]), error="current edition")

    # Stored rows are independently reverified at publication and read, even after staging.
    snapshot = json.loads(snapshots[1, 1].read_text())
    snapshot["rows"][0]["surface"] = "corrupt storage"
    with sqlite3.connect(db) as writer:
        writer.execute("UPDATE edition_versions SET snapshot=? WHERE stream=? AND seq=?",
                       (json.dumps(snapshot).encode(), snapshot["checkpoint"]["stream"],
                        f"{snapshot['checkpoint']['sequence']:020d}"))
    client.call("read", db, error="native checkpoint")
    manifest = json.loads(manifests[1].read_text())
    manifest["edition"] = 42
    client.call("publish", db, save(temp/"altered-storage-manifest", manifest), error="native checkpoint")
    client.call("stage", db, snapshots[1, 1])
    assert client.call("read", db)["members"] == bulk[1]
    # A full target snapshot can skip an unavailable predecessor.
    empty_db = temp/"new-subscriber.sqlite"
    client.call("stage", empty_db, deltas[0, 2], error="base unavailable")
    for p in range(3):
        client.call("stage", empty_db, snapshots[p, 2])
    client.call("publish", empty_db, manifests[2])
    assert client.call("read", empty_db)["members"] == bulk[2]

    # IndexSnapshot fingerprints key order. An equivalent reordered wire array
    # must still yield the same deterministic public snapshot on every reader.
    doc = grounded_names("ensemble", [("First", "lead"), ("Second", "lead")], catalog="credit")
    native = query(binary, save(temp/"ensemble", dict(stream="program", sequence=1, document=doc)))
    ordered = save(temp/"ordered", native)
    shuffled = save(temp/"shuffled", {**native, "rows": list(reversed(native["rows"]))})
    group = save(temp/"ensemble-manifest", client.call("manifest", 3, ordered))
    client.call("stage", db, shuffled)
    client.call("publish", db, group)
    assert client.call("read", db)["members"] == [native]
    client.call("stage", db, ordered)
    assert client.call("read", db)["members"] == [native]

    private = Client(binary, "staff")
    private.call("publish", db, manifests[1], error="scope differs")
    private.call("read", db, error="unavailable")
    private_manifest = private.call("manifest", 1, snapshots[2, 1])
    waiting = private.call("publish", db, save(temp/"staff", private_manifest))
    assert waiting["missing"] == [withheld["checkpoint"]] and waiting["current_edition"] is None
    client.call("manifest", 8, snapshots[2, 0], snapshots[2, 1], error="duplicate edition member")
    invalid = json.loads(manifests[1].read_text())
    invalid["members"].append(withheld["checkpoint"])
    client.call("publish", db, save(temp/"duplicate-member", invalid), error="duplicate edition member")
    for field in ("collection", "source", "audience"):
        invalid = json.loads(manifests[1].read_text())
        invalid["scope"][field] = " "
        client.call("publish", db, save(temp/"empty-scope", invalid), error="nonempty")
    empty = save(temp/"empty", client.call("manifest", 2**64-1))
    assert client.call("publish", db, empty)["status"] == "published"
    assert client.call("read", db)["members"] == []
    conflict = json.loads(empty.read_text())
    conflict["members"] = [withheld["checkpoint"]]
    client.call("publish", db, save(temp/"conflict", conflict), error="conflicts")
    assert client.call("publish", db, manifests[1])["current_edition"] == 2**64-1


def integrity_cases(client, temp, baseline, snapshots, manifests, bulk):
    db = temp/"integrity.sqlite"
    shutil.copyfile(baseline, db)
    native = json.loads(snapshots[0, 0].read_text())
    original = json.loads(native["checkpoint"]["stream"])
    for part, value, error in [(2, "other-document", "posting document"),
                               (3, ["credit", "understudy"], "posting catalog record")]:
        altered = copy.deepcopy(native)
        stream = original.copy()
        stream[part] = value
        altered["checkpoint"]["stream"] = json.dumps(stream, separators=(",", ":"))
        receipt = save(temp/"relabeled", altered)
        client.call("stage", db, receipt, error=error)
        client.call("manifest", 7, receipt, error=error)

    # A second encoding of one tuple at the same sequence must not introduce
    # another stream or bypass the guard on the published member.
    alternate = copy.deepcopy(native)
    alternate["checkpoint"]["stream"] = json.dumps(original)  # extra spaces
    path = save(temp/"ambiguous-stream", alternate)
    client.call("stage", db, path, error="noncanonical")
    client.call("manifest", 7, path, snapshots[0, 0], error="noncanonical")
    client.call("evict", db, save(temp/"ambiguous-checkpoint", alternate["checkpoint"]),
                error="noncanonical")
    manifest = json.loads(manifests[0].read_text())
    ambiguous = {**manifest, "members": [*manifest["members"], alternate["checkpoint"]]}
    client.call("publish", db, save(temp/"ambiguous-manifest", ambiguous), error="noncanonical")
    assert client.call("read", db)["members"] == bulk[0]

    # Only the persisted manifest body changes. It still parses, and every
    # remaining checkpoint still names a verified native snapshot.
    subset = {**manifest, "members": manifest["members"][:1]}
    with sqlite3.connect(db) as writer:
        writer.execute("UPDATE published_editions SET manifest=?", (json.dumps(subset).encode(),))
    client.call("read", db, 0, error="manifest digest")
    client.call("read", db, error="manifest digest")
    dropped = save(temp/"still-pinned", manifest["members"][1])
    client.call("evict", db, dropped, error="manifest digest")
    client.call("publish", db, manifests[0], error="manifest digest")
    client.call("publish", db, save(temp/"truncated-manifest", subset), error="manifest digest")
    client.call("publish", db, manifests[1], error="manifest digest")
    with sqlite3.connect(db) as reader:
        assert reader.execute("SELECT count(*) FROM edition_versions WHERE snapshot IS NOT NULL").fetchone() == (3,)
        assert reader.execute("SELECT count(*) FROM published_editions").fetchone() == (1,)

    with sqlite3.connect(db) as writer:
        writer.execute("DELETE FROM published_editions")
    client.call("read", db, error="manifest body missing")
    client.call("evict", db, dropped, error="manifest body missing")
    client.call("publish", db, manifests[0], error="manifest body missing")


def run(binary):
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        client = Client(binary)
        snapshots, deltas, manifests, bulk = prepare(binary, directory, client)
        base = schedules(client, directory, snapshots, deltas, manifests, bulk)
        failure_cases(binary, client, directory, base, snapshots, deltas, manifests, bulk)
        integrity_cases(client, directory, base, snapshots, manifests, bulk)
    print("90 schedules, 540 process-isolated reads matched complete bulk editions; recovery checks passed")


def measure(binary, members, rows):
    """Offline subscriber cost including process startup, local disk and verification."""
    with tempfile.TemporaryDirectory() as name:
        directory = Path(name)
        client = Client(binary)
        versions = [[], []]
        changes = []
        manifests = []
        for i in range(members):
            previous = None
            for edition in range(2):
                occurrences = [(f"Performer {j}", "alternate" if edition and j == 0 else "cast")
                               for j in range(rows)]
                doc = grounded_names(f"program-{i}", occurrences, catalog="credit")
                source = save(directory/f"doc-{i}-{edition}",
                              dict(stream="program", sequence=edition, document=doc))
                versions[edition].append(save(directory/f"receipt-{i}-{edition}", query(binary, source)))
                if previous:
                    changes.append(save(directory/f"delta-{i}", query(binary, source, prior=previous)))
                previous = source
        for edition, snapshots in enumerate(versions):
            manifest = client.call("manifest", edition, *snapshots)
            manifests.append(save(directory/f"edition-{edition}", manifest))
        db = directory/"catalog.sqlite"
        for snapshot in versions[0]:
            client.call("stage", db, snapshot)
        client.call("publish", db, manifests[0])
        start = time.perf_counter()
        for change in changes:
            client.call("stage", db, change)
        staged = time.perf_counter()
        client.call("publish", db, manifests[1])
        published = time.perf_counter()
        with sqlite3.connect(db) as reader:
            retained, = reader.execute("SELECT SUM(length(snapshot)) FROM edition_versions").fetchone()
        fresh = directory/"fresh.sqlite"
        catch_up = time.perf_counter()
        for snapshot in versions[1]:
            client.call("stage", fresh, snapshot)
        client.call("publish", fresh, manifests[1])
        recovered = time.perf_counter()
        latest = client.call("read", db)
        assert client.call("read", fresh) == latest
        assert sum(len(member["rows"]) for member in latest["members"]) == members * rows
        return dict(members=members, rows_per_member=rows, two_editions_snapshot_bytes=retained,
                    database_bytes=db.stat().st_size, first_delta_to_release_s=round(published-start, 3),
                    last_member_staged_to_release_s=round(published-staged, 3),
                    full_snapshot_catchup_s=round(recovered-catch_up, 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anno", type=Path, help="use an existing anno executable")
    parser.add_argument("--measure", action="store_true",
                        help="measure two synthetic corpus sizes instead of running the correctness gate")
    args = parser.parse_args()
    binary = (args.anno or build_cli()).resolve()
    if args.measure:
        for size in (4, 24):
            print(json.dumps(measure(binary, size, 250)), flush=True)
    else:
        run(binary)
