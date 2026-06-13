"""Tests for the anno_py binding layer.

Run after `maturin develop --uv` in crates/anno-py.

These exercise the offline (pattern + heuristic) path only; email detection
via RegexNER is deterministic (confidence 0.98), so it anchors the assertions.
"""

import anno_py


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
