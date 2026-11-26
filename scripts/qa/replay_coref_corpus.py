"""Replay a named corpus witness against the primary, exact marked-link contract."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import coref_corpus as corpus

SCOPE = "coreference_primary_marked_links_v1"
PASS_SIGNATURE = "primary_marked_comparison_matched"
FAIL_SIGNATURE = r"(?:nested_endpoint_unavailable|nested_wrong_identity|wrong_marked_links:(?:A|B|both|neither))"
LOCAL_INPUTS = "Set ANNO_COREF_SOURCES (local JSON path map), ANNO_COREF_RUNNER and ANNO_COREF_MODEL_DIR."
OPAQUE_ID = {
    "dutch-openboek": r"[0-9a-f]{24}", "french-litbankfr": r"[0-9a-f]{24}",
    "gap": r"[0-9a-f]{16}", "gum": r"[0-9a-f]{64}", "litbank": r"[0-9a-f]{16}",
    "phrase-detectives": r"pd3-[0-9a-f]{24}", "winogender": r"[0-9a-f]{20}",
}


class MissingLocalInputs(FileNotFoundError):
    """The corpus or native producer has not been provisioned."""


class NoMarkedComparison(ValueError):
    """A source case offers no scoreable primary decision."""


def family_for_case(root, case):
    # Check the transferable name without requiring restricted datasets on this host.
    if not isinstance(case, str):
        raise ValueError("expected a corpus name and opaque case ID")
    name, _, case_id = case.partition("/")
    if name not in corpus.READERS or name not in OPAQUE_ID or not re.fullmatch(OPAQUE_ID[name], case_id):
        raise ValueError("expected <corpus>/<opaque case ID>")
    return name


def _local(name):
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        raise MissingLocalInputs(LOCAL_INPUTS)
    return Path(value)


def _case(root, case):
    name = family_for_case(root, case)
    try:
        args = SimpleNamespace(source=None, sources_file=_local("ANNO_COREF_SOURCES"))
        try:
            path = corpus.source_path(args, name)
        except corpus.CorpusError as exc:
            raise MissingLocalInputs(LOCAL_INPUTS) from exc
        if not path.exists():
            raise MissingLocalInputs(LOCAL_INPUTS)
        item, = corpus.select_cases(corpus.READERS[name](path), case_id=case.partition("/")[2])
    except FileNotFoundError as exc:
        raise MissingLocalInputs(LOCAL_INPUTS) from exc
    if item["gold"] == "unscored" or (item["kind"] == "choice" and not ({"A", "B"} & item["spans"].keys())):
        raise NoMarkedComparison("This case has no scored primary marked comparison.")
    return name, item


def _producer():
    runner, model = _local("ANNO_COREF_RUNNER"), _local("ANNO_COREF_MODEL_DIR")
    if not runner.is_file() or not os.access(runner, os.X_OK) or not model.is_dir():
        raise MissingLocalInputs(LOCAL_INPUTS)
    return runner, model


def describe(root, case):
    name, item = _case(root, case)
    local = Path(__file__).resolve().parent
    target = root / "scripts/qa"
    # Include transitive span validation and cluster indexing performed by score().
    code = {}
    for fn in (describe, corpus.select_cases, corpus.READERS[name], corpus.model_spans,
               corpus.index_partition, corpus.source_sha256):
        path = Path(inspect.getfile(fn)).resolve()
        relative = path.relative_to(local)
        contents = path.read_bytes()
        if (target / relative).read_bytes() != contents:
            raise ValueError("coreference replay code differs from selected checkout")
        code[str(relative)] = hashlib.sha256(contents).hexdigest()
    registry = json.loads((target / "coref_corpora/sources.json").read_bytes())
    try:
        producer = corpus.producer_identity(*_producer())
    except FileNotFoundError as exc:
        raise MissingLocalInputs(LOCAL_INPUTS) from exc
    oracle = {
        "scope": SCOPE, "case": case, "source_revision": registry[name]["revision"],
        "selection_and_score": code,
        "marked_reference": {k: item[k] for k in ("kind", "spans", "gold", "boundaries", "aliases") if k in item},
        "entry_point": "fcoref_word_spans", "boundary_policy": "default", "batch_size": 1,
        "producer": producer,
    }
    return {
        "family": name, "source_sha256": corpus.source_sha256(item["text"]),
        "oracle_sha256": hashlib.sha256(json.dumps(oracle, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def replay(root, case):
    _, item = _case(root, case)
    runner, model = _producer()
    row, = corpus.run_native([item], runner, model, batch_size=1)
    primary = row["boundaries"]["default"]
    if primary["matches_marked_gold"] is None:
        raise NoMarkedComparison("This case has no scored primary marked comparison.")
    observed = primary["observed"]
    if primary["matches_marked_gold"]:
        # "Neither" agrees only on the supplied candidates, independently of the
        # target's referential status or possible identity links elsewhere.
        outcome, signature = "pass", PASS_SIGNATURE
    elif item["kind"] == "nested":
        outcome = "fail"
        signature = "nested_endpoint_unavailable" if observed == "unresolved" else "nested_wrong_identity"
    else:
        outcome, signature = "fail", "wrong_marked_links:" + observed
    return {"outcome": outcome, "signature": signature, "producer_sha256": corpus.path_sha(runner)}
