"""Line-delimited JSON bridge for the optional maverick-coref Python runtime."""
import json
import os
import sys
from contextlib import redirect_stdout


def resolve(model, tokenizer, text, mentions, singletons):
    # Maverick's raw-string preprocessor assumes one space between sentences.
    # Supply tokenized sentences and use the original document's character offsets.
    sentences = [[t for t in s if not t.is_space] for s in tokenizer(text).sents]
    sentences = [s for s in sentences if s]
    tokens = [t for s in sentences for t in s]
    predefined = None
    if mentions is not None:
        starts = {t.idx: i for i, t in enumerate(tokens)}
        ends = {t.idx + len(t): i for i, t in enumerate(tokens)}
        predefined = []
        for start, end in mentions:
            if start >= end or start not in starts or end not in ends:
                raise ValueError(f"mention [{start}, {end}) must cover complete tokens")
            predefined.append((starts[start], ends[end]))
    if not tokens or predefined == []:
        return []
    output = model.predict([[t.text for t in s] for s in sentences],
                           singletons=singletons, predefined_mentions=predefined)
    allowed = set(predefined) if predefined is not None else None
    clusters = []
    for cluster in output['clusters_token_offsets']:
        spans = []
        for first, last in cluster:
            if not 0 <= first <= last < len(tokens):
                raise ValueError(f"invalid Maverick token range [{first}, {last}]")
            if allowed is not None and (first, last) not in allowed:
                raise ValueError("Maverick returned a mention outside the supplied spans")
            spans.append((tokens[first].idx, tokens[last].idx + len(tokens[last])))
        clusters.append(spans)
    return clusters


def load_model(checkpoint):
    from maverick import Maverick
    if not checkpoint.endswith(".safetensors"):
        return Maverick(hf_name_or_path=checkpoint, device="cpu")

    from maverick.models.model_mes import Maverick_mes
    from safetensors.torch import load_file

    class PreparedMES(Maverick):
        def __init__(self):
            # Fixed upstream MES/DeBERTa-large inference architecture.
            # Configuration and Python classes never come from the checkpoint.
            self.device = "cpu"
            self.model = Maverick_mes(huggingface_model_name="microsoft/deberta-v3-large",
                                      freeze_encoder=False, span_representation="concat_start_end")
            self.model.load_state_dict(load_file(checkpoint), strict=True)
            self.model.eval()
            self.tokenizer = self.__get_model_tokenizer__()

    return PreparedMES()


def main():
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        if os.environ.get("ANNO_NO_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Applies to both the encoder and Lightning's legacy checkpoint loader.
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        import spacy
        model = load_model(sys.argv[1])
        tokenizer = spacy.blank("en")
        tokenizer.add_pipe("sentencizer")
        print('"ready"', file=protocol, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                spans = resolve(model, tokenizer, **request)
                response = {"clusters": spans}
            except Exception as error:
                response = {"error": str(error)}
            print(json.dumps(response), file=protocol, flush=True)


if __name__ == "__main__":
    main()
