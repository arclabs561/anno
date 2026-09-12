"""Tests for the anno_py binding layer.

Run after `maturin develop --uv` in crates/anno-py.

The default path exercises offline pattern + heuristic extraction; email
detection is deterministic (confidence 0.98), so it anchors the assertions.
Set ANNO_PY_TEST_MODELS=1 with an ONNX wheel to opt into cached real-model
coverage without permitting downloads.
"""

import os

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
    with pytest.raises(ValueError, match="does not accept"):
        anno_py.Extractor(backend="offline", model="some-model")


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

    with pytest.raises(RuntimeError, match="ONNX-enabled wheel"):
        anno_py.Extractor(backend=backend)


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
    text = "Barack Obama visited Paris."
    model = anno_py.Extractor(backend=backend, **kwargs)
    first = model.extract(text)
    warm = model.extract(text)

    assert first
    assert all(text[entity.start : entity.end] == entity.text for entity in first)
    assert [(e.text, e.label, e.start, e.end) for e in first] == [
        (e.text, e.label, e.start, e.end) for e in warm
    ]
