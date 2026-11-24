#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exercise the reader example across process boundaries, including its saved document."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ["cargo", "run", "--quiet", "--locked", "-p", "anno", "--no-default-features",
           "--example", "review_mention", "--"]


def run(*args, archive=None, check=True):
    result = subprocess.run(
        [*COMMAND, *args], cwd=ROOT, input=archive, text=True, capture_output=True,
    )
    if check and result.returncode:
        raise RuntimeError(f"review_mention {args[0]} failed:\n{result.stderr}")
    return result


def catalog_for(doc, signal_id):
    track, = [track for track in doc["tracks"].values()
              if any(member["signal_id"] == signal_id for member in track["signals"])]
    identity = doc["identities"][str(track["identity_id"])]
    return identity["kb_name"], identity["kb_id"]


def main():
    preview = json.loads(run("preview").stdout)
    plan = preview["proposal"]
    assert plan["previous"]["kb_id"] == "musician" and plan["kb_id"] == "clerk"
    assert plan["affected_tracks"] == [plan["trigger"]]
    reason = "I checked the signature in the quay log."

    for action, kind, current in [("apply", "applied", "clerk"), ("defer", "deferred", "musician")]:
        session = run("review", archive=f"{action}\n{reason}\n")
        archive = session.stdout
        record = json.loads(archive)
        evidence, decision, doc = record["evidence"], record["decision"], record["document"]
        assert record["reason"] == reason and decision["kind"] == kind
        assert evidence == preview["evidence"] and evidence["doc_id"] == doc["id"]
        assert preview["source_text"] == doc["text"] and doc["text"] in session.stderr
        assert hashlib.sha256(doc["text"].encode()).hexdigest() == evidence["source_sha256"]
        assert doc["text"][evidence["start"]:evidence["end"]] == evidence["quote"] == "Ora Venn"
        assert (evidence["start"], evidence["end"]) == (36, 44)
        if action == "apply":
            assert decision["receipt"]["assignment"] == plan
        else:
            assert decision["proposal"] == plan and "receipt" not in decision
        selected = evidence["signal_id"]
        other, = [s["id"] for s in doc["signals"] if s["surface"] == "Ora Venn" and s["id"] != selected]
        assert catalog_for(doc, selected) == ("quay-catalog", current)
        assert catalog_for(doc, other) == ("quay-catalog", "musician")
        shown = run("show", archive=archive).stdout
        assert f"Decision {kind}" in shown and f"Reader reason {reason}" in shown
        assert f"Target in saved document quay-catalog/{current}" in shown
        assert "Prior target quay-catalog/musician" in shown and "Proposed target quay-catalog/clerk" in shown

        if action == "apply":
            # An in-place correction must carry the same record ID before and after.
            changed = json.loads(archive)
            receipt = changed["decision"]["receipt"]
            assert receipt["assignment"]["updates_shared_identity"]
            receipt["assignment"]["previous"]["identity_id"] = receipt["identity_id"] + 1
            assert run("show", archive=json.dumps(changed), check=False).returncode != 0

        # Substitute the homonym's signal ID while retaining the selected occurrence's range.
        evidence["signal_id"] = other
        assert run("show", archive=json.dumps(record), check=False).returncode != 0
    unfinished = run("review", archive="apply\n", check=False)
    assert unfinished.returncode != 0 and not unfinished.stdout.strip()
    print("Reopened applied and deferred decisions, checked both namesakes, and rejected a swapped anchor.")


if __name__ == "__main__":
    main()
