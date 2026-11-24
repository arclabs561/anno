"""Replay bounded, agent-authored grounded observations through the offline cross-doc CLI.

Document revisions are submitted in order. 'after' selects a prefix, and source ranges
are half-open Unicode code-point offsets in that document's latest edition there.
Assertions compare report groups, which can also aggregate by qualified catalog record.
'absent' means no matching supplied signal, 'unresolved' means a supplied signal has
no reported membership, and 'resolved' includes singleton groups. Separation requires
two positive memberships. Inputs already carry their observed signals and tracks.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import time


def _fields(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required) <= obj.keys() <= set(required) | set(optional):
        raise ValueError(f"expected fields {', '.join(required)} with only declared optional fields")


def _items(value, maximum, minimum=0):
    if type(value) is not list or not minimum <= len(value) <= maximum:
        raise ValueError(f"expected a list with {minimum} to {maximum} entries")
    return value


def _int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"expected integer in {low}..{high}")
    return value


def _name(value, maximum=96):
    if not isinstance(value, str) or not 1 <= len(value) <= maximum or not value.isprintable() or not value.strip():
        raise ValueError("expected a bounded, nonblank printable identifier")
    return value


def _number(value, low, high):
    if type(value) not in (float, int) or not low <= value <= high or not math.isfinite(value):
        raise ValueError("expected a finite number within the allowed range")
    return value


def _record(value, *, qualified=False):
    catalog, accession = _items(value, 2, 2)
    _name(catalog)
    _name(accession)
    if qualified and catalog.strip().lower() == "unknown":
        raise ValueError("use null to assert absence of a qualified catalog record")
    return catalog, accession


def _wire(doc):
    _fields(doc, ("id", "text", "signals", "tracks"))
    ident, text = _name(doc["id"]), doc["text"]
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("expected source text of at most 4096 code points")
    text.encode("utf-8")  # Reject unpaired surrogates, which cannot be submitted as Rust text.
    signals, tracks, identities = [], {}, {}
    anchors = set()
    for pos, signal in enumerate(_items(doc["signals"], 64)):
        _fields(signal, ("start", "end", "label"))
        start = _int(signal["start"], 0, len(text))
        end = _int(signal["end"], start + 1, len(text))
        label = _name(signal["label"])
        anchor = (start, end, label)
        if anchor in anchors:
            raise ValueError("duplicate source span and label cannot identify distinct signals")
        anchors.add(anchor)
        # Sparse IDs exercise the CLI's documented flat-position mention coordinates.
        signals.append({"id": 10 + 2 * pos, "location": {"Text": {"start": start, "end": end}},
                        "surface": text[start:end], "label": label, "confidence": 1.0,
                        "modality": "Symbolic", "negated": False})
    assigned = set()
    for idx, track in enumerate(_items(doc["tracks"], 64)):
        _fields(track, ("name", "members"), ("record", "label", "embedding"))
        name = _name(track["name"])
        members = []
        for member in _items(track["members"], len(signals), 1):
            _int(member, 0, len(signals) - 1)
            if member in assigned:
                raise ValueError("a supplied signal cannot occur twice in track membership")
            assigned.add(member)
            members.append({"signal_id": signals[member]["id"], "position": len(members)})
        row = {"id": idx, "signals": members, "canonical_surface": name, "cluster_confidence": 1.0}
        if "label" in track:
            row["entity_type"] = _name(track["label"])
        if "embedding" in track:
            row["embedding"] = [_number(n, -1e10, 1e10) for n in _items(track["embedding"], 64, 1)]
        if "record" in track:
            kb, accession = _record(track["record"])
            row["identity_id"] = idx
            identities[str(idx)] = {"id": idx, "canonical_name": name, "entity_type": row.get("entity_type"),
                                    "kb_name": kb, "kb_id": accession, "confidence": 1.0, "aliases": [],
                                    "source": {"KnowledgeBase": {"kb_name": kb, "kb_id": accession}}}
        tracks[str(idx)] = row
    return {"id": ident, "text": text, "signals": signals, "tracks": tracks, "identities": identities}


def _editions(wires, after):
    return {doc["id"]: (revision, doc) for revision, doc in enumerate(wires[:after], 1)}


def _locate(ref, editions):
    _items(ref, 4, 3)
    ident = _name(ref[0])
    if ident not in editions:
        raise ValueError("referenced document has not been submitted at this checkpoint")
    revision, doc = editions[ident]
    start = _int(ref[1], 0, len(doc["text"]))
    end = _int(ref[2], start + 1, len(doc["text"]))
    label = _name(ref[3]) if len(ref) == 4 else None
    positions = [i for i, signal in enumerate(doc["signals"])
                 if signal["location"]["Text"] == {"start": start, "end": end}
                 and (label is None or signal["label"] == label)]
    if len(positions) > 1:
        raise ValueError("ambiguous source span; supply its label as a fourth reference item")
    return revision, (ident, positions[0]) if positions else None


def validate_case(case):
    """Validate all checkpoints and source references before execution. Return grounded documents."""
    _fields(case, ("kind", "id", "documents", "assertions"), ("threshold", "require_type_match"))
    if case["kind"] != "cross_doc":
        raise ValueError("expected cross_doc case")
    _name(case["id"])
    _number(case.get("threshold", 0.6), 0, 1)
    if type(case.get("require_type_match", False)) is not bool:
        raise ValueError("require_type_match must be a boolean")
    wires = [_wire(doc) for doc in _items(case["documents"], 24, 1)]
    if sum(len(doc["signals"]) for doc in wires) > 256:
        raise ValueError("a case may supply at most 256 signals across its revisions")
    checkpoints = set()
    for rule in _items(case["assertions"], 256, 1):
        _fields(rule, ("kind", "after", "mentions"), ("record",))
        kind = rule["kind"]
        if kind not in ("same", "separate", "resolved", "unresolved", "absent", "record"):
            raise ValueError("unknown cross_doc assertion kind")
        if (kind == "record") != ("record" in rule):
            raise ValueError("only record assertions require an expected record (or null)")
        if rule.get("record") is not None:
            _record(rule["record"], qualified=True)
        after = _int(rule["after"], 1, len(wires))
        checkpoints.add(after)
        count = 2 if kind in ("same", "separate") else 1
        refs = _items(rule["mentions"], count, count)
        editions = _editions(wires, after)
        coords = [_locate(ref, editions)[1] for ref in refs]
        if count == 2 and (refs[0] == refs[1] or (coords[0] is not None and coords[0] == coords[1])):
            raise ValueError("pair assertions require two distinct occurrences")
    if len(checkpoints) > 8:
        raise ValueError("a case may select at most eight revision checkpoints")
    return wires


def _membership(report, editions):
    if not isinstance(report, dict) or not isinstance(report.get("metadata"), dict):
        raise ValueError("malformed cross-doc report")
    meta = report["metadata"]
    count = sum(len(doc["signals"]) for _, doc in editions.values())
    for key, expected in (("documents_processed", len(editions)), ("total_entities", count)):
        if type(meta.get(key)) is not int or meta[key] != expected:
            raise ValueError(f"CLI report disagrees with submitted {key}")
    membership = {}
    for group, row in enumerate(_items(report.get("clusters"), count)):
        if not isinstance(row, dict):
            raise ValueError("malformed cluster")
        kb, acc = row.get("kb_name"), row.get("kb_id")
        for field in (kb, acc):
            if field is not None:
                _name(field)
        record = [kb, acc] if kb and kb.strip().lower() != "unknown" and acc else None
        documents = set()
        for mention in _items(row.get("mentions"), count, 1):
            ident, position = _items(mention, 2, 2)
            if _name(ident) not in editions:
                raise ValueError("unknown document in reported membership")
            _int(position, 0, len(editions[ident][1]["signals"]) - 1)
            if (ident, position) in membership:
                raise ValueError("duplicate reported membership for a source occurrence")
            documents.add(ident)
            membership[ident, position] = (group, record)
        reported = [_name(ident) for ident in _items(row.get("documents"), len(editions), 1)]
        if set(reported) != documents or len(reported) != len(documents):
            raise ValueError("reported cluster documents disagree with its mentions")
    return membership


def _observe(rule, editions, membership):
    observations, groups = [], []
    for ref in rule["mentions"]:
        revision, occurrence = _locate(ref, editions)
        member = membership.get(occurrence)
        state = "absent" if occurrence is None else "unresolved" if member is None else "resolved"
        observations.append({"input_revision": revision, "state": state, "record": member[1] if member else None})
        groups.append(member[0] if member else None)
    kind = rule["kind"]
    comparison = None
    if kind in ("same", "separate"):
        if None not in groups:
            comparison = groups[0] == groups[1]
        passed = comparison is not None and comparison == (kind == "same")
    elif kind == "record":
        passed = observations[0]["state"] == "resolved" and observations[0]["record"] == rule["record"]
    else:
        passed = observations[0]["state"] == kind
    return {**rule, "passed": passed, "observed": observations, "same_report_group": comparison}


def run_case(case: dict, binary: Path) -> dict:
    """Replay explicit grounded data with the selected executable. Contract or CLI errors raise ValueError."""
    wires = validate_case(case)
    path = Path(binary).resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("select an executable anno binary with eval support")
    args = [str(path), "cross-doc", "--stdin", "--format", "json",
            "--threshold", str(case.get("threshold", 0.6))]
    if case.get("require_type_match"):
        args.append("--require-type-match")
    env = {**os.environ, "ANNO_NO_DOWNLOADS": "1", "HF_HUB_OFFLINE": "1"}
    lines = [json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n" for doc in wires]
    deadline = time.monotonic() + 30
    snapshots = {}
    for after in sorted({rule["after"] for rule in case["assertions"]}):
        try:
            result = subprocess.run(args, input="".join(lines[:after]).encode(), capture_output=True,
                                    env=env, timeout=min(10, max(0, deadline - time.monotonic())))
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"could not complete offline cross-doc checkpoint {after}") from exc
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            raise ValueError(f"invalid offline cross-doc execution at checkpoint {after} (exit {result.returncode})")
        try:
            report = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid cross-doc JSON at checkpoint {after}") from exc
        editions = _editions(wires, after)
        snapshots[after] = editions, _membership(report, editions)
    return {"checks": [_observe(rule, *snapshots[rule["after"]]) for rule in case["assertions"]]}
