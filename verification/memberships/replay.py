"""Publish native membership snapshots/deltas and exercise an indexed SQLite reader."""
import argparse
import hashlib
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from statistics import median
import sqlite3
import subprocess
import tempfile
from time import perf_counter_ns


def document(id, name, rev="1", catalog=None):
    doc = {"id": id, "text": name, "signals": [], "tracks": {}, "identities": {}}
    if name:
        doc["signals"].append({"id": 0, "surface": name, "label": "PER", "confidence": 1,
            "location": {"Text": {"start": 0, "end": len(name)}}, "modality": "Symbolic", "negated": False})
        doc["tracks"]["0"] = {"id": 0, "canonical_surface": name, "entity_type": "PER",
            "signals": [{"signal_id": 0, "position": 0}], "cluster_confidence": 1}
        if catalog:
            ns, code = catalog
            doc["identities"]["0"] = {"id": 0, "canonical_name": name, "entity_type": "PER",
                "kb_name": ns, "kb_id": code, "confidence": 1, "aliases": [],
                "source": {"KnowledgeBase": {"kb_name": ns, "kb_id": code}}}
            doc["tracks"]["0"]["identity_id"] = 0
    return {"source_revision": rev, "document": doc}


def edition(seq, inputs):
    return dict(stream="register", sequence=seq, documents=inputs, threshold=0.5, require_type_match=True)


def capture(cli, head, base=None, neighbors=None):
    with tempfile.TemporaryDirectory() as scratch:
        args = [cli, "query", "memberships", "-"]
        if base is not None:
            path = Path(scratch) / "previous.json"
            path.write_text(json.dumps(base))
            args += ["--since", str(path)]
        if neighbors:
            args += ["--neighbors", neighbors, "0"]
        result = subprocess.run(args, input=json.dumps(head), text=True, capture_output=True, timeout=60)
        if result.returncode:
            raise ValueError(result.stderr)
        return json.loads(result.stdout)


class Reader:
    """The local CLI delivers intact output over a pipe to this SQLite pilot."""

    def __init__(self, snapshot):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
            CREATE TABLE posting (key TEXT PRIMARY KEY, component TEXT, value TEXT);
            CREATE INDEX members ON posting(component);
            CREATE TABLE head (checkpoint TEXT NOT NULL);
        """)
        with self.db:
            self.write(snapshot["rows"])
            self.db.execute("INSERT INTO head VALUES (?)", (json.dumps(snapshot["checkpoint"]),))
        self.cached = {}

    def write(self, rows):
        self.db.executemany("INSERT INTO posting VALUES (?, ?, ?)",
            [(json.dumps(key), row["component"], json.dumps(row)) for key, row in rows])

    def apply(self, delta):
        dirty = set()
        changed = set()
        with self.db:
            current = json.loads(self.db.execute("SELECT checkpoint FROM head").fetchone()[0])
            if current == delta["target"]:
                return
            if current != delta["base"]:
                raise ValueError("wrong predecessor")
            for key, row in delta["removed"]:
                self.db.execute("DELETE FROM posting WHERE key=?", (json.dumps(key),))
            self.write(delta["upserted"])
            self.db.execute("UPDATE head SET checkpoint=?", (json.dumps(delta["target"]),))
            for key, row in delta["removed"] + delta["upserted"]:
                changed.add(tuple(key))
                if row["component"] is not None:
                    dirty.add(row["component"])
        self.cached = {key: value for key, value in self.cached.items()
                       if key not in changed and value[0] not in dirty}

    def query(self, key):
        key = tuple(key)
        if key in self.cached:
            return self.cached[key][1]
        posting = self.db.execute("SELECT component, value FROM posting WHERE key=?", (json.dumps(key),)).fetchone()
        result = []
        component = None
        if posting is not None:
            component, value = posting
            direct = {tuple(edge["track"]): edge for edge in json.loads(value)["direct"]}
            for other, raw in self.db.execute("SELECT key, value FROM posting WHERE component=?", (component,)):
                peer = tuple(json.loads(other))
                if peer != key:
                    row = json.loads(raw)
                    result.append({"key": list(peer), "direct_similarity": direct.get(peer, {}).get("similarity"),
                                   "direct_basis": direct.get(peer, {}).get("basis"),
                                   "surface": row["surface"], "catalog_record": row["catalog_record"]})
        result.sort(key=lambda peer: peer["key"])
        self.cached[key] = (component, result)
        return result


def check(cli):
    docs = {name: document(name, surface) for name, surface in
            [("a","Jori"),("b","Jori"),("c","Moss"),("d","Moss"),("side","Oxton")]}
    docs["catalog"] = document("catalog", "Jori", catalog=("folio","17"))
    previous = reader = None
    all_keys = set()
    # Old groups, single bridge, redundant support, corrected single bridge, full split,
    # disjoint append, identical-text new source edition, and complete withdrawal.
    edits = [{}, {"x":document("x","Jori Moss")}, {"y":document("y","Moss Jori")},
             {"x":document("x","Kite Lake","2")}, {"y":None}, {"z":document("z","Ḧaru")},
             {"a":document("a","Jori","2")}, {id:None for id in [*docs,"x","z"]}]
    calls = 0
    for seq, changes in enumerate(edits):
        for id, new in changes.items():
            if new is None:
                docs.pop(id)
            else:
                docs[id] = new
        current = edition(seq, list(docs.values()))
        full = capture(cli,current)
        reverse = capture(cli,edition(seq,list(reversed(current["documents"]))))
        assert full == reverse
        if previous is None:
            reader = Reader(full)
        else:
            delta = capture(cli,current,previous)
            reader.apply(delta)
            reader.apply(delta)
        rebuilt = Reader(full)
        current_keys = {k[0]:k for k,_ in full["rows"]}
        all_keys.update(tuple(k) for k in current_keys.values())
        for key, row in full["rows"]:
            source = docs[key[0]]
            text = source["document"]["text"]
            assert key == [key[0], hashlib.sha256(text.encode()).hexdigest(), source["source_revision"], "0"]
            signal, = row["signals"]
            assert signal["location"] == {"Text": {"start": 0, "end": len(text)}} and signal["surface"] == text
        # Independent set intersection on the submitted source names gives the fixed
        # unanchored edges for this panel; catalogue records do not enter comparison.
        neighbors = defaultdict(set)
        direct = {}
        for a,b in combinations(docs,2):
            if a=="catalog" or b=="catalog":
                continue
            left,right = [set(docs[i]["document"]["text"].lower().split()) for i in (a,b)]
            score = len(left & right)/len(left | right)
            if score >= .5:
                neighbors[a].add(b)
                neighbors[b].add(a)
                direct[a,b] = direct[b,a] = score
        for key in all_keys:
            found = reader.query(key)
            assert found == rebuilt.query(key)
            if list(key) != current_keys.get(key[0]):
                assert found == []
                continue
            source = key[0]
            seen,todo = {source},[source]
            while todo:
                new = neighbors[todo.pop()] - seen
                seen.update(new)
                todo.extend(new)
            assert {r["key"][0]:r["direct_similarity"] for r in found} == {
                peer:direct.get((source,peer)) for peer in seen-{source}}
            live = capture(cli,current,neighbors=source)
            assert live["track"] == list(key) and live["neighbors"] == found
            calls += 1
        if "catalog" in current_keys:
            assert dict((k[0],v) for k,v in full["rows"])["catalog"]["catalog_record"] == {
                "catalog":"folio","accession":"17"}
        previous = current
    for bad, base in [(edition(10, [document("a","Jori"),document("a","Moss")]), None),
                      (dict(edition(10, []), threshold=.75), previous),
                      (edition(1, []), previous)]:
        try:
            capture(cli,bad,base)
        except ValueError:
            continue
        raise AssertionError("ambiguous source or incompatible predecessor was accepted")
    print(f"{len(edits)} snapshots, {calls} current native neighbor queries, cached and bulk rows agree")


def cost(cli):
    for n in [32,128]:
        docs = [document(str(i),f"group{i//4}") for i in range(n)]
        old,new = edition(1, docs),edition(2,docs+[document("bridge","group0 group1")])
        baseline = capture(cli,old)
        timings=defaultdict(list)
        for _ in range(5):
            reader=Reader(baseline)
            start=perf_counter_ns()
            capture(cli,new)
            split=perf_counter_ns()
            delta=capture(cli,new,old)
            applied=perf_counter_ns()
            reader.apply(delta)
            end=perf_counter_ns()
            for key,ns in [("one_full_ms",split-start),("two_full_plus_diff_ms",applied-split),
                           ("apply_ms",end-applied)]:
                timings[key].append(ns/1e6)
        print(json.dumps({"base_docs":n,"candidate_pairs_current":(n+1)*n//2,
                          "removed":len(delta["removed"]),"upserted":len(delta["upserted"]),
                          **{k:round(median(v),3) for k,v in timings.items()}}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("cli")
    parser.add_argument("--cost",action="store_true")
    args=parser.parse_args()
    check(args.cli)
    if args.cost:
        cost(args.cli)
