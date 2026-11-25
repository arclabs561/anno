# /// script
# requires-python = ">=3.10"
# dependencies = ["gliner==0.2.30", "torch>=2.6", "sentencepiece"]
# ///
"""Extract NuNER Zero-4k mentions for `anno dataset import --format jsonl`.

    uv run scripts/nuner_4k_import.py letter.txt person location > mentions.jsonl
    anno dataset import --format jsonl --input mentions.jsonl --source-text letter.txt --output imported.jsonl
"""

import argparse
import json
from pathlib import Path

CHECKPOINT = "numind/NuNER_Zero-4k"
REVISION = "7a7cd8d65af2572c297054dbaf8f25c0d46da55d"


def extract(model, text, labels, threshold):
    # Check both training-time word limits and Longformer's encoder limit.
    # The upstream data collator can otherwise truncate a document silently.
    processor = model.data_processor
    words = [word for word, _, _ in processor.words_splitter(text)]
    if len(words) > model.config.max_len:
        raise ValueError(f"Document exceeds the checkpoint's {model.config.max_len}-word limit")
    prompt, _ = processor.prepare_inputs([words], [list(labels)])
    tokens = processor.transformer_tokenizer(prompt[0], is_split_into_words=True, truncation=False)
    if len(tokens["input_ids"]) > 4096:
        raise ValueError("Prompt and document together exceed Longformer's 4096-token window")

    raw = model.predict_entities(text, list(labels), threshold=threshold)
    rows = []
    for word in sorted(raw, key=lambda item: (item["start"], item["end"])):
        start, end = word["start"], word["end"]
        if not (0 <= start < end <= len(text)) or word["text"] != text[start:end]:
            raise ValueError(f"Invalid source span [{start}, {end})")
        label = labels[word["label"]]
        previous = rows[-1] if rows else None
        if previous and previous["type"] == label and (
            start == previous["end"]
            or (start == previous["end"] + 1 and text[previous["end"]].isspace())
        ):
            previous.update(end=end, text=text[previous["start"]:end])
            # The checkpoint scores the individual words. It supplies no score
            # for the conjunction that defines this merged mention.
            previous.pop("confidence", None)
        else:
            rows.append({
                "start": start,
                "end": end,
                "type": label,
                "text": word["text"],
                "confidence": word["score"],
                "source": f"{CHECKPOINT}@{REVISION}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    labels = {label.lower(): label for label in args.labels}
    if "" in labels or len(labels) != len(args.labels):
        parser.error("Supply nonempty, distinct labels (case-insensitive)")
    text = args.source.read_bytes().decode("utf-8")
    if not text.strip():
        return

    from gliner import GLiNER
    from huggingface_hub import snapshot_download

    checkpoint = snapshot_download(
        CHECKPOINT,
        revision=REVISION,
        allow_patterns=["gliner_config.json", "pytorch_model.bin"],
    )
    model = GLiNER.from_pretrained(checkpoint)
    model.eval()
    for row in extract(model, text, labels, args.threshold):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
