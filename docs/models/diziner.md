# Preparing a DiZiNER schema review offline

The paper's authors published the disagreement analysis implementation, benchmark schemas, supervisor prompts and orchestration source. Anno can use the published analysis on local pilot annotations to find disagreements for schema review. This step generates candidate review material. The complete paper method uses multiple LLM annotators and a supervisor which revises the instructions. [Published paper](https://aclanthology.org/2026.acl-long.795/) · [Pinned implementation](https://github.com/SiunKim/diziner-ner/tree/2577f8ce7f06f0554e88b1931f9e0c9626c24f17)

For the English CoNLL schema, the preparation script selects 20 deterministic rows from the pinned CrossNER development file and discards its gold tags. It runs the local Anno BERT and heuristic models on the same text, projects predictions onto the four published schema labels and records rows that cannot be represented by exact flat BIO tags. It then runs DiZiNER's official `analyze_ner_disagreement_dataset` function with its default majority-vote settings and its 80th-percentile uncertainty cutoff. The result includes the official schema, final task goal, raw analysis summary, returned thresholds, each review block and the actual conflicting labels. The script verifies the imported upstream code by hash and performs local inference with downloads disabled.

```sh
git clone https://github.com/SiunKim/diziner-ner /tmp/diziner
git -C /tmp/diziner checkout --detach 2577f8ce7f06f0554e88b1931f9e0c9626c24f17
curl -fL https://raw.githubusercontent.com/zliucr/CrossNER/2e7ba2a7798c961e3f29fbc51252c5a8d40224bf/ner_data/conll2003/dev.txt -o /tmp/conll-dev.txt
export BERT_ONNX_MODEL_PATH=/path/to/local/bert-base-NER-onnx
uv run scripts/qa/diziner_pilot.py --upstream /tmp/diziner --anno target/debug/anno \
  --dev /tmp/conll-dev.txt --output /tmp/diziner-review.json
```

For the recorded run, BERT used `protectai/bert-base-NER-onnx` at revision `84d28ab2bc28401b4353b26033eb043c8c8a0739`. Of 20 pilot sentences, 17 supported exact projection and three had model spans that split a corpus token. The official analyzer gave equal weights to the two annotators. At the selected percentile the uncertainty cutoff was zero, so the analyzer returned all 17 sentences as blocks. Eleven blocks contained a real label disagreement and were retained for review. One assigned different types to Hampshire, and another disagreed on whether a player's rank numeral belonged to the name. These are concrete prompts for a reviewer to clarify team names and name boundaries using the corpus conventions. Disagreement alone does not establish the correct label. This pilot produced no supervisor-generated guidelines or held-out accuracy estimate.
