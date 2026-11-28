# AgCNER character offsets

Anno now presents all **59,776** AgCNER passages to the model exactly as the publisher encoded them. All **206,992** labelled character spans agree with an independent reading of the three Figshare files. The previous reader inserted 3,849,517 spaces and preserved only 105 original label-and-offset pairs. The generic heuristic scored **zero exact matches** with either reader, so AgCNER does not support a claim of agricultural Chinese NER coverage by that backend. [Replay receipt](agcner-source-coordinates.json)

| Original split | Passages | Characters | Gold spans | Original spans recovered before | Recovered after | Heuristic predictions before → after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 47,839 | 3,127,977 | 165,686 | 88 | 165,686 | 12,971 → 1,153 |
| Development | 5,963 | 394,843 | 20,782 | 8 | 20,782 | 1,770 → 150 |
| Test | 5,974 | 386,473 | 20,524 | 9 | 20,524 | 1,548 → 155 |

The release places one Unicode character and its BIO label on each nonblank line. Blank lines separate passages. The repaired reader concatenates the characters and computes half-open entity offsets in Unicode scalar values. The original BIO files and existing token-array JSONL caches produced identical source hashes, labelled span fingerprints and scores in every passage. The official release contains zero whitespace tokens and zero non-BMP tokens. Regression tests cover a literal space, a tab and a supplementary-plane character. [Publisher collection](https://doi.org/10.6084/m9.figshare.c.6807873)

## Reproduce

Use eval-enabled executables built from the baseline revision `1b6fcc97df321e1614bdc94f00c9b18ab36bb49a` and the repaired revision. The replay reads the three original files, verifies their SHA-256 digests, and runs the heuristic through `anno dataset eval --cached-only --reports`. Only aggregate results are written to the receipt.

```sh
python scripts/qa/agcner_source_replay.py \
  --before "$ANNO_BEFORE" --after "$ANNO_AFTER" \
  --source-dir "$PWD/agcner-originals" --fetch \
  --output agcner-source-coordinates.json
```

For an offline run, put `train.bio`, `dev.bio` and `test.bio` in the source directory and omit `--fetch`. A normal cold `anno dataset eval --dataset AgCNER --model heuristic` selects the official test file. [Replay script](../../scripts/qa/agcner_source_replay.py)
