"""Local raw-text adapter for the released CorPipe 26 one-stage predictor."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def ground(text, prediction):
    from udapi import Document

    doc = Document()
    doc.from_conllu_string(prediction)
    clusters = []
    for entity in doc.coref_entities:
        mentions = []
        for mention in entity.mentions:
            tokens = []
            for node in mention.words:
                token = {"sentence": node.root.sent_id, "word_id": str(node.ord)}
                if node.is_empty():
                    token["alignment"] = "empty_node"
                else:
                    surface = node.multiword_token or node
                    bounds = surface.misc["TokenRange"].split(":")
                    if len(bounds) != 2:
                        raise ValueError(f"No source token range for {node.address()}")
                    start, end = map(int, bounds)
                    if not 0 <= start < end <= len(text):
                        raise ValueError(
                            f"Out-of-bounds source token at {node.address()}"
                        )
                    token.update(
                        start=start,
                        end=end,
                        text=text[start:end],
                        alignment="multiword_token" if node.multiword_token else "word",
                    )
                tokens.append(token)
            spans = [t for t in tokens if "start" in t]
            item = {"head_word_id": str(mention.head.ord), "tokens": tokens}
            if spans:
                # One-stage CorPipe emits contiguous overt mentions. Multiple UD
                # words may share the surface range of a contraction.
                start, end = (
                    min(t["start"] for t in spans),
                    max(t["end"] for t in spans),
                )
                item.update(start=start, end=end, text=text[start:end])
            mentions.append(item)
        clusters.append({"id": entity.eid, "mentions": mentions})
    return clusters


def main(predictor, checkpoint, tokenizer_model, output):
    from ufal.udpipe import Model, Pipeline, ProcessingError

    predictor, checkpoint, tokenizer_model = map(
        Path, (predictor, checkpoint, tokenizer_model)
    )
    if not predictor.is_file() or not (checkpoint / "model.pt").is_file():
        raise ValueError(
            "Supply the one-stage predictor script and a local CorPipe checkpoint directory"
        )
    tokenizer = Model.load(str(tokenizer_model))
    if tokenizer is None:
        raise ValueError(f"Cannot load UDPipe model {tokenizer_model}")
    # Read bytes to preserve CRLF and other original source offsets.
    text = sys.stdin.buffer.read().decode("utf-8")
    raw = text.encode("utf-8")
    metadata = {
        "model": "CorPipe 26 one-stage",
        "checkpoint": str(checkpoint),
        "tokenizer_model": str(tokenizer_model),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "offset_unit": "unicode_scalar",
        "source_length": len(text),
    }
    prediction = ""
    if text.strip():
        parser = Pipeline(
            tokenizer, "tokenizer=ranges", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
        )
        error = ProcessingError()
        conllu = parser.process(text, error)
        if error.occurred():
            raise ValueError(error.message)
        with tempfile.TemporaryDirectory(prefix="anno-corpipe-") as tmp:
            directory = Path(tmp)
            source = directory / "source.conllu"
            source.write_text(conllu, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(predictor.resolve()),
                    "--load",
                    str(checkpoint.resolve()),
                    "--exp",
                    tmp,
                    "--epoch",
                    "0",
                    "--test",
                    str(source),
                ],
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            if result.stdout:
                print(result.stdout, end="", file=sys.stderr)
            result.check_returncode()
            if "Truncating a long sentence during prediction" in result.stdout:
                raise ValueError(
                    "CorPipe truncated a sentence beyond its scoring window"
                )
            prediction = (directory / "source.00.conllu").read_text(encoding="utf-8")
    if output == "conllu":
        sys.stdout.write(prediction)
    else:
        metadata["clusters"] = ground(text, prediction) if prediction else []
        print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
