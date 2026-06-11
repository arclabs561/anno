# anno-py

Python bindings for [anno](https://github.com/arclabs561/anno).

This build uses the pattern + heuristic backends: no model downloads,
works offline. Offsets are character offsets, so `text[e.start:e.end] == e.text`.

## Install (development)

```sh
cd crates/anno-py
uv venv && maturin develop --uv
```

## Usage

```python
import anno_py

ents = anno_py.extract("Contact Jane Doe at jane.doe@example.com.")
for e in ents:
    print(e.text, e.label, e.start, e.end, e.confidence)
```

For repeated calls, reuse an extractor: `ex = anno_py.Extractor()`, then `ex.extract(text)`.

## Tests

```sh
pytest tests/
```
