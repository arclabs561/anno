"""Exercise installed anno_py wheels against explicitly supplied local checkpoints."""

import json
import os
import subprocess

import anno_py
import pytest

CASES = [
    (
        "bert",
        "ANNO_PY_TEST_BERT_DIR",
        "onnx",
        "🌍 Angela Merkel visited Berlin.",
        {},
        {"Berlin"},
    ),
    (
        "gliner",
        "ANNO_PY_TEST_GLINER_EDGE",
        "onnx",
        "Élodie: Email john.smith@acme.com or call +1 415 555 0199.",
        {"labels": ["email address", "phone number"], "threshold": 0.3},
        {"john.smith@acme.com", "+1 415 555 0199"},
    ),
    (
        "adaptner",
        "ADAPTNER_MODEL_DIR",
        "onnx",
        "خارطة فلسطين تبين طرق التهجير التي سلكها الفلسطينيون أثناء حرب 1948 ( النكبة ) .",
        {},
        {"1948"},
    ),
    (
        "gliner2-pii",
        "GLINER2_PII_MODEL_DIR",
        "fastino",
        "Élodie: Email john.smith@acme.com or call +1 415 555 0199.",
        {"labels": ["email"]},
        {"john.smith@acme.com"},
    ),
    (
        "glinerel-72m",
        "ANNO_GLINEREL_MODEL_DIR",
        None,
        "🙂 Ada Lovelace was born in London.",
        {},
        {"Ada Lovelace", "London"},
    ),
    (
        "kniv",
        "KNIV_MODEL_DIR",
        None,
        "🙂 Ada Lovelace was born in London.",
        {},
        {"Ada Lovelace", "London"},
    ),
    (
        "otter-cross",
        "OTTER_CROSS_MODEL_DIR",
        None,
        "🙂 Angela Merkel besuchte Berlin.",
        {"labels": ["person", "location"]},
        {"Angela Merkel"},
    ),
    (
        "otter-bi",
        "OTTER_BI_MODEL_DIR",
        None,
        "🙂 Angela Merkel besuchte Berlin.",
        {"labels": ["person", "location"]},
        {"Angela Merkel"},
    ),
]


@pytest.mark.parametrize(
    ("backend", "model_env", "feature", "text", "options", "expected"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_local_checkpoint(
    backend, model_env, feature, text, options, expected, monkeypatch
):
    if os.environ.get("ANNO_PY_TEST_MODELS") != "1":
        pytest.skip("set ANNO_PY_TEST_MODELS=1 for local checkpoint tests")
    if feature and not getattr(anno_py, f"__{feature}_enabled__"):
        pytest.skip(f"wheel lacks the {feature} feature")
    model = os.environ.get(model_env)
    if not model:
        pytest.skip(f"set {model_env} to the prepared checkpoint")
    monkeypatch.setenv("ANNO_NO_DOWNLOADS", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    extractor = anno_py.Extractor(backend=backend, model=model, **options)
    first = extractor.extract(text)
    repeated = extractor.extract(text)
    del extractor  # Release the resident worker before the CLI loads the checkpoint.
    for entities in (first, repeated):
        surfaces = {entity.text for entity in entities}
        assert expected <= surfaces
        if backend == "gliner2-pii":
            assert surfaces == expected
        assert all(text[e.start : e.end] == e.text for e in entities)
    cli = os.environ.get("ANNO_PY_TEST_CLI")
    if cli and feature is None:
        args = [cli, "extract", "--model", backend, "--format", "json", "--text", text]
        if "labels" in options:
            args.extend(["--extract-types", ",".join(options["labels"])])
        result = json.loads(subprocess.check_output(args, timeout=300))
        rust = sorted(
            (e["type"], e["start"], e["end"], e["text"]) for e in result["entities"]
        )
        assert rust == sorted((e.label, e.start, e.end, e.text) for e in first)
    assert [(e.label, e.start, e.end) for e in first] == [
        (e.label, e.start, e.end) for e in repeated
    ]
