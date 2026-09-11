# Constituency fixture

`benepar_en3.json` is actual output of `scripts/parse_benepar.py` using
`benepar==0.2.0`, the `benepar_en3` model and spaCy's blank English tokenizer.
The input sentence pair was authored for this test, not taken from a treebank:

```text
Mira builds small robots. She smiles.
```

Regenerate with:

```sh
uv run --locked scripts/parse_benepar.py --download \
  --text 'Mira builds small robots. She smiles.' > testdata/constituency/benepar_en3.json
```

The fixture checks import compatibility, offsets, nesting and rendering. It is
not a parsing-accuracy benchmark. Model weights and training data are not
included. Upstream implementation: <https://github.com/nikitakit/self-attentive-parser>.
