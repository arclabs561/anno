"""Tests for the anno_py binding layer.

Run after `maturin develop --uv` in crates/anno-py.

The default path exercises offline pattern + heuristic extraction; email
detection is deterministic (confidence 0.98), so it anchors the assertions.
Set ANNO_PY_TEST_MODELS=1 with an ONNX wheel to opt into cached real-model
coverage without permitting downloads.
"""

import json
import math
import os
from pathlib import Path

import anno_py
import pytest


def test_extract_email():
    text = "Contact us at support@example.com for help."
    ents = anno_py.extract(text)
    emails = [e for e in ents if e.label == "EMAIL"]
    assert len(emails) == 1
    e = emails[0]
    assert e.text == "support@example.com"
    # char offsets index directly into the Python string
    assert text[e.start : e.end] == e.text
    assert 0.0 <= e.confidence <= 1.0


def test_char_offsets_with_emoji():
    # Non-ASCII prefix shifts byte offsets away from char offsets; the
    # invariant text[start:end] == text only holds for char offsets.
    text = "\U0001f4e7\U0001f4e7 email: jane.doe@company.org"
    ents = anno_py.Extractor().extract(text)
    emails = [e for e in ents if e.label == "EMAIL"]
    assert len(emails) == 1
    e = emails[0]
    assert e.text == "jane.doe@company.org"
    assert text[e.start : e.end] == e.text


def test_extractor_reuse_and_empty_input():
    ex = anno_py.Extractor()
    assert ex.extract("") == []
    # second call on the same instance works (no consumed state)
    assert any(
        e.label == "EMAIL" for e in ex.extract("ping admin@example.org today")
    )


def test_repr():
    ents = anno_py.extract("mail me: a@b.io")
    assert ents, "expected at least the email entity"
    r = repr(ents[0])
    assert r.startswith("Entity(") and "label=" in r


def test_offline_backend_is_explicit_and_rejects_model_options():
    ex = anno_py.Extractor(backend="offline")
    assert any(e.label == "EMAIL" for e in ex.extract("admin@example.org"))
    with pytest.raises(ValueError, match="does not accept") as error:
        anno_py.Extractor(backend="offline", model="some-model")
    assert error.type is ValueError


def test_invalid_backend_and_gliner_options():
    with pytest.raises(ValueError, match="backend must be one of"):
        anno_py.Extractor(backend="unknown")
    with pytest.raises(ValueError, match="threshold"):
        anno_py.Extractor(backend="gliner", threshold=1.1)
    with pytest.raises(ValueError, match="labels"):
        anno_py.Extractor(backend="gliner", labels=[])
    with pytest.raises(ValueError, match="does not support zero-shot"):
        anno_py.Extractor(backend="bert", labels=["person"])
    with pytest.raises(ValueError, match="does not support zero-shot"):
        anno_py.Extractor(backend="bert", threshold=0.5)


@pytest.mark.parametrize("backend", ["bert", "gliner"])
def test_model_backends_require_onnx_in_offline_wheel(backend):
    if getattr(anno_py, "__onnx_enabled__", False):
        pytest.skip("feature-unavailable assertion applies only to an offline wheel")

    with pytest.raises(RuntimeError, match="ONNX-enabled wheel") as error:
        anno_py.Extractor(backend=backend)
    assert error.type is RuntimeError


def test_fastino_capability_gate():
    assert isinstance(anno_py.__fastino_enabled__, bool)
    assert callable(anno_py.FastinoExtractor)
    if not anno_py.__fastino_enabled__:
        with pytest.raises(RuntimeError, match="Fastino-enabled source build"):
            anno_py.FastinoExtractor()


def test_fastino_cache_miss_is_explicit(monkeypatch):
    if not getattr(anno_py, "__fastino_enabled__", False):
        pytest.skip("requires a Fastino-enabled source build")

    # This deliberately nonexistent repository must fail from the local cache
    # while downloads are forbidden; it must not fall back to another model.
    monkeypatch.setenv("ANNO_NO_DOWNLOADS", "1")
    with pytest.raises(RuntimeError, match="ANNO_NO_DOWNLOADS"):
        anno_py.FastinoExtractor(model="anno-py-test/nonexistent-fastino-model")


@pytest.mark.parametrize(
    ("backend", "kwargs"),
    [
        ("bert", {}),
        ("gliner", {"labels": ["person", "location"], "threshold": 0.3}),
    ],
)
def test_cached_model_extractors_are_repeatable(monkeypatch, backend, kwargs):
    """Run only when explicitly opted into cached real-model coverage."""
    if not getattr(anno_py, "__onnx_enabled__", False):
        pytest.skip("requires an ONNX-enabled wheel")
    if os.environ.get("ANNO_PY_TEST_MODELS") != "1":
        pytest.skip("set ANNO_PY_TEST_MODELS=1 to test cached real models")

    # Constructor and extraction must fail rather than download on a cache miss.
    monkeypatch.setenv("ANNO_NO_DOWNLOADS", "1")
    text = "🌍 Barack Obama visited Paris."
    model = anno_py.Extractor(backend=backend, **kwargs)
    first = model.extract(text)
    warm = model.extract(text)

    assert first
    assert all(text[entity.start : entity.end] == entity.text for entity in first)
    assert [(e.text, e.label, e.start, e.end) for e in first] == [
        (e.text, e.label, e.start, e.end) for e in warm
    ]


def test_cached_fastino_binding_is_repeatable(monkeypatch):
    """Exercise the pinned Fastino Python binding against cached assets."""
    if not getattr(anno_py, "__fastino_enabled__", False):
        pytest.skip("requires a Fastino-enabled source build")
    if os.environ.get("ANNO_PY_TEST_MODELS") != "1":
        pytest.skip("set ANNO_PY_TEST_MODELS=1 to test the cached Fastino model")

    monkeypatch.setenv("ANNO_NO_DOWNLOADS", "1")
    model = anno_py.FastinoExtractor()
    text = "🌍 Acme Corp signed a deal with Globex in Paris."
    labels = ["organization", "location"]
    first = model.extract(text, labels=labels, threshold=0.5)
    warm = model.extract(text, labels=labels, threshold=0.5)

    # Normalize Rust-owned output before comparison: entity ordering is an
    # implementation detail, but text, label, character offsets and score are
    # the binding contract. A tolerance avoids platform-level float noise.
    def normalized(entities):
        return sorted((e.text, e.label, e.start, e.end) for e in entities)

    assert first
    assert normalized(first) == normalized(warm)
    assert sorted(entity.confidence for entity in first) == pytest.approx(
        sorted(entity.confidence for entity in warm), abs=1e-5
    )
    assert all(text[entity.start : entity.end] == entity.text for entity in first)
    assert any(entity.text == "Acme Corp" for entity in first)
    assert any(entity.text == "Paris" for entity in first)

    classes = model.classify(
        "This product is wonderful, I love it.",
        labels=["positive", "negative", "neutral"],
    )
    assert classes[0].label == "positive"
    assert {result.label for result in classes} == {"positive", "negative", "neutral"}
    assert all(math.isfinite(result.probability) for result in classes)
    assert all(
        earlier.probability >= later.probability
        for earlier, later in zip(classes, classes[1:])
    )
    assert sum(result.probability for result in classes) == pytest.approx(1.0, abs=1e-5)

    with pytest.raises(ValueError, match="threshold"):
        model.extract(text, labels=labels, threshold=float("nan"))
    with pytest.raises(ValueError, match="labels"):
        model.extract(text, labels=[])
    with pytest.raises(ValueError, match="labels"):
        model.classify(text, labels=[""])


def test_cached_fastino_binding_matches_rust_reference(monkeypatch):
    """Compare the installed binding with the feature-gated Rust oracle."""
    if not getattr(anno_py, "__fastino_enabled__", False):
        pytest.skip("requires a Fastino-enabled source build")
    if os.environ.get("ANNO_PY_TEST_MODELS") != "1":
        pytest.skip("set ANNO_PY_TEST_MODELS=1 to test the cached Fastino model")
    reference_path = os.environ.get("ANNO_PY_FASTINO_REFERENCE")
    if reference_path is None:
        pytest.skip("set ANNO_PY_FASTINO_REFERENCE to a Rust Fastino oracle JSON file")

    reference = json.loads(Path(reference_path).read_text())
    assert reference["model"] == "jugaadsrl/gliner2-multi-v1-onnx"
    monkeypatch.setenv("ANNO_NO_DOWNLOADS", "1")
    model = anno_py.FastinoExtractor()

    ner = reference["ner"]
    actual_entities = model.extract(
        ner["text"], labels=ner["labels"], threshold=ner["threshold"]
    )
    expected_entities = sorted(
        (item["text"], item["label"], item["start"], item["end"], item["confidence"])
        for item in ner["entities"]
    )
    actual_normalized = sorted(
        (entity.text, entity.label, entity.start, entity.end, entity.confidence)
        for entity in actual_entities
    )
    assert [row[:4] for row in actual_normalized] == [row[:4] for row in expected_entities]
    assert [row[4] for row in actual_normalized] == pytest.approx(
        [row[4] for row in expected_entities], abs=1e-5
    )
    assert all(
        ner["text"][entity.start : entity.end] == entity.text
        for entity in actual_entities
    )

    classification = reference["classification"]
    actual_classes = model.classify(
        classification["text"], labels=classification["labels"]
    )
    assert [result.label for result in actual_classes] == [
        item["label"] for item in classification["results"]
    ]
    assert [result.probability for result in actual_classes] == pytest.approx(
        [item["probability"] for item in classification["results"]], abs=1e-5
    )
