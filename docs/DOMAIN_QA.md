# Compare extraction on ten labeled domains

Use this panel to ask whether a change recovers exact entity mentions across subjects and genres. Each eligible input contributes one independently annotated mention. The discovery and reserved halves have the same per-domain budget and disjoint input texts. Keep the reserved fixtures sealed until the candidate and the decision rule are frozen.

| Groups | Source and annotation |
|---|---|
| AI, literature, music, politics, science | Five CrossNER topics share an encyclopedia genre. Only general person, organization and location labels are eligible. [CrossNER test files](https://github.com/zliucr/CrossNER/tree/2e7ba2a7798c961e3f29fbc51252c5a8d40224bf/ner_data) |
| Reuters news | Person, organization and location labels in the CoNLL-2003 test data distributed with CrossNER. [Snapshot](https://github.com/zliucr/CrossNER/tree/2e7ba2a7798c961e3f29fbc51252c5a8d40224bf/ner_data/conll2003) |
| Social posts | WNUT-17 person, corporation and location annotations. [WNUT test file](https://github.com/leondz/emerging_entities_17/blob/e52a2d2a71ac1a051ca2d532eef28653c5c02603/emerging.test.annotated) |
| Flight and assistant queries | ATIS city, state and country name slots, including directional variants. SNIPS city, state and country slots. Both are rendered as LOC. [JointBERT test files](https://github.com/monologg/JointBERT/tree/00324f60dda4a491b2913bc3065ae93966caea3f/data) |
| Contracts | Source-aligned CUAD Parties answers with company suffixes. Each contract contributes at most one window with up to 250 characters of surrounding context per side. [CUAD release](https://github.com/TheAtticusProject/cuad/tree/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2) |

The [source catalog](../scripts/qa/domain-sources.json) lists exact upstream files, revisions and hashes. Keep datasets, generated fixtures and predictions local. Upstream data terms apply to reuse of the texts.

## Prepare the fixtures

The fetch command is optional and makes network requests. To work offline, place the exact source files in the data directory under the filenames from the catalog. For CUAD, use the archive's `test.json` as `cuad.json`. Preparation always checks the uncompressed file hashes.

```sh
data=.generated/anno-qa/domain-sources
panel=.generated/anno-qa/domain-panel
python3 scripts/qa/prepare_domain_fixtures.py fetch "$data"
just qa-prepare-domains "$data" "$panel"
```

The default selects up to 32 discovery and 32 reserved inputs per domain. Set the two trailing recipe arguments to change the per-split budget and seed. Preparation requires a new output directory. To use only selected local sources, run `python3 scripts/qa/prepare_domain_fixtures.py prepare "$data" "$panel" --domain atis --domain snips`.

`manifest.json` records the source families, SHA-256 pins, seed, eligible records, unique eligible texts after exclusions, actual split sizes and any shortfall per domain. Cases are ranked deterministically within each group. Exact duplicate inputs are allocated once across the panel. Changing the source selection can therefore change membership. Token/tag sources reconstruct text with one space between tokens and preserve case. The resulting offsets refer to that exact UTF-8 source interpreted as Unicode characters. Different sentences can come from the same article.

## Run a paired comparison

Set `BASELINE` and `CANDIDATE` to the executables under comparison. On a new fixture, the baseline normally misses some anchors. The checker still saves its source-bound results when it exits 1 for an unmet assertion. Keep exit 2 for invalid input or execution failures.

```sh
fixture="$panel/discovery.json"
checker=scripts/qa/check_span_assertions.py
base_rc=0
python3 "$checker" --fixture "$fixture" --binary "$BASELINE" --model stacked \
    --save-predictions "$panel/base.jsonl" > "$panel/base-report.json" || base_rc=$?
if [ "$base_rc" -gt 1 ]; then exit "$base_rc"; fi

python3 "$checker" --fixture "$fixture" --binary "$CANDIDATE" --model stacked \
    --baseline-predictions "$panel/base.jsonl" > "$panel/paired-report.json"
```

The report lists the baseline and candidate outcome for each assertion. Group its findings by the domain before the slash in each `id`; report the actual denominator and paired gains and losses. Keep separate prediction/report paths when repeating the commands on `reserved.json`. The normal [round tracer](ADVERSARIAL_QA.md) can record the commands and their input/output digests.

The measured quantity is exact recall on selected labeled anchors. Other mentions and predictions are unjudged. Inputs with no eligible anchor are excluded, so the panel cannot estimate precision or false alarms on empty documents. A change that fails to improve these anchors may still improve a specific extraction pattern. Freeze fresh source groups before testing a repair informed by these results.
