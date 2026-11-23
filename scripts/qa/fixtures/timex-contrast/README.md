# Temporal contrast panel

The 34 English excerpts contain 16 annotated time expressions in 13 documents and 21 documents
without a time expression. The public news test split has a time expression in every sentence.
Use this panel alongside that split to measure false alarms on names, titles and date-free prose.

An evaluator selected and labeled the excerpts before seeing the compared predictions. Fourteen
examples are original paired probes. The remaining twenty quote the sources below, with whitespace
normalized. The annotations are ours. The panel deliberately includes ambiguous words, so its
false-alarm rate applies to these excerpts. [Frozen input](rows.jsonl)

Score exact zero-based, half-open Unicode character spans against each row's text. Count a prediction
of “Saturday” inside the gold “Saturday nights” as one extra span and one missed span. Also count
the number of gold-empty documents with at least one temporal prediction.

| Source key | Text and source terms |
|---|---|
| `constitution` | US Constitution transcript, public domain original. [US Constitution](https://www.archives.gov/founding-docs/constitution-transcript) |
| `declaration` | US Declaration of Independence transcript, public domain original. [US Declaration of Independence](https://www.archives.gov/founding-docs/declaration-transcript) |
| `pep20` | PEP 20 by Tim Peters; placed in the public domain. [The Zen of Python](https://peps.python.org/pep-0020/) |
| `alice` | Lewis Carroll, Alice's Adventures in Wonderland (1865), public domain in the US. [Alice’s Adventures in Wonderland](https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/1d493f4f5487e3cd29b9bbd9d2b4d1bccccee9f8/11-0.txt) |
| `crusoe` | Daniel Defoe, Robinson Crusoe (1719), public domain in the US. [Robinson Crusoe](https://raw.githubusercontent.com/GITenberg/The-Life-and-Adventures-of-Robinson-Crusoe_521/8de290556fe8c5c6d41e5547bead2a5032722f44/521-0.txt) |
| `faraday` | Michael Faraday, The Chemical History of a Candle (1861), public domain in the US. [The Chemical History of a Candle](https://raw.githubusercontent.com/GITenberg/The-Chemical-History-of-a-Candle_14474/dbffcef2ef521638dab7e24e22195282ec7ed84a/14474-8.txt) |

The prefixes `authored-` identify the original examples. The SHA-256 of the frozen JSONL is
`c1718c1749c528234fe2007f903bfa48bad3e1da5ff1f25e9b1fd5d5554ec4f9`.

On the frozen panel the native pattern backend matched 4 of 16 annotated spans with 4 predictions.
GLiNER Multitask matched 9 with 12 predictions. One of its extra predictions fell in a date-free
document: Defoe’s character Friday. It also marked “July” inside “July 4, 1776” and “delivery date”
inside a document containing a separate date. Pattern raised no alarm in the 21 date-free texts.
The GLiNER evaluation used the FP32 ONNX graph and the label `time expression` at threshold 0.5.
[Pinned model](https://huggingface.co/onnx-community/gliner-multitask-large-v0.5/tree/c1b022a51e6bc8535702b6b37fc38a3ed7fe00c6)

Run the native temporal scorer from the repository root with an eval-enabled Anno CLI. Use an
isolated cache so the panel gets its own input digest and history.

```sh
panel_cache=$(mktemp -d)
mkdir -p "$panel_cache/datasets"
cp scripts/qa/fixtures/timex-contrast/rows.jsonl "$panel_cache/datasets/TimexRecognitionSentenceOriginal.cache"
ANNO_CACHE_DIR="$panel_cache" ANNO_NO_DOWNLOADS=1 ANNO_S3_CACHE=0 \
  anno sampler --strategy random run --task temporal --fixed-backend pattern \
    --fixed-datasets TimexRecognitionSentenceOriginal --require-cached --max-examples 34 \
    --runs 1 --seed-base 42 --decisions-file "$panel_cache/decisions.jsonl"
```

To compare another installed temporal backend, use the same file and seed in a fresh cache.

For independently annotated news and task-oriented specification cases, prepare the two
[source-pinned temporal panels](../../temporal_sources/README.md).
