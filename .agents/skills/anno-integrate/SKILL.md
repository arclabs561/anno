---
name: anno-integrate
description: Integrate Anno in Rust or Python, project extracted spans back to source documents, or preserve annotation identity and provenance across formats.
---

# Integrate Anno

Use the docs from the checkout or release your application targets. Start with the [documentation index](https://github.com/arclabs561/anno/blob/main/docs/README.md) if you need to locate a guide.

| What does the application need? | Where to start |
| --- | --- |
| Extract repeatedly in Rust or Python | See “Library usage” in `docs/QUICKSTART.md` for Rust and `crates/anno-py/README.md` for Python. Both show how to reuse an extractor. The Python bulletin example ties each occurrence to a source edition. |
| Highlight the original after cleanup, normalization or HTML rendering | Read “Source text and offsets” in `docs/CONTRACT.md` and the HTML walkthrough in `docs/EXTRACTION_WORKFLOWS.md`. Use the saved map to project analyzed characters to original byte fragments. Check the original digest and whether each fragment is exact enough for the intended use. |
| Preserve a discontinuous mention or an OCR bounding box | Read `docs/LOCATION.md` before choosing a text projection. Keep individual ranges for a discontinuous mention and retain geometry when the reader needs it. |
| Exchange brat, CoNLL, JSONL or RDF with another tool | Use the format-specific examples in `docs/EXTRACTION_WORKFLOWS.md`. Check a round trip against the exact source edition, including repeated surfaces and gaps. |
| Review a catalog assignment or compare two grounded snapshots | Follow the quay-log examples in `docs/QUICKSTART.md`. They show how to select one occurrence, record a decision and inspect the resulting change in catalog results. |
