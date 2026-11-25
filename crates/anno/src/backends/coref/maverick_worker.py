"""Line-delimited JSON bridge for the optional maverick-coref Python runtime."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
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


def prepared_checkpoint(checkpoint):
    model_ids = {f"sapienzanlp/maverick-mes-{name}": name for name in ("ontonotes", "litbank", "preco")}
    if checkpoint in model_ids:
        cache = os.environ.get("ANNO_CACHE_DIR")
        if not cache:
            base = Path.home() / "Library/Caches" if sys.platform == "darwin" else Path(
                os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            cache = base / "anno"
        name = model_ids[checkpoint]
        path = Path(cache) / "models" / "maverick" / f"maverick-{name}.safetensors"
        if not path.is_file():
            raise ValueError(f"prepare {checkpoint} with scripts/download_maverick_weights.py --model {name}")
    else:
        path = Path(checkpoint)
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError("Maverick needs a prepared MES safetensors file or a prepared official model ID")
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as archive:
        if (archive.metadata() or {}).get("anno_architecture") != "maverick-mes-deberta-v3-large-v1":
            raise ValueError("expected prepared Maverick MES DeBERTa-v3-large safetensors")
    return path


def verify_encoder(pins, *, cache_dir=None, directory=None):
    from huggingface_hub import hf_hub_download
    offline = os.environ.get("ANNO_NO_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    for name, expected in pins["ENCODER_FILES"].items():
        source = Path(hf_hub_download(pins["ENCODER"], name, revision=pins["ENCODER_REVISION"],
                                      cache_dir=cache_dir, local_files_only=offline))
        if directory is not None:
            # AutoModel and AutoTokenizer inspect optional sibling files. Give them
            # only the four reviewed files, even if the Hub cache also contains others.
            destination = directory / name
            try:
                os.link(source.resolve(), destination)
            except OSError:
                shutil.copyfile(source, destination)
            source = destination
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"encoder digest mismatch for {name}")


def load_model(checkpoint, pins):
    checkpoint = prepared_checkpoint(checkpoint)
    from maverick import Maverick
    from maverick.models.model_mes import Maverick_mes
    from safetensors.torch import load_file

    class PreparedMES(Maverick):
        def __init__(self, encoder_directory):
            # Fixed upstream MES/DeBERTa-large inference architecture.
            # Configuration and Python classes never come from the checkpoint.
            self.device = "cpu"
            self.model = Maverick_mes(huggingface_model_name=str(encoder_directory),
                                      freeze_encoder=False, span_representation="concat_start_end")
            self.model.load_state_dict(load_file(checkpoint), strict=True)
            self.model.eval()
            self.tokenizer = self.__get_model_tokenizer__()

    with tempfile.TemporaryDirectory() as temporary:
        encoder_directory = Path(temporary)
        verify_encoder(pins, directory=encoder_directory)
        return PreparedMES(encoder_directory)


def main():
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        if os.environ.get("ANNO_NO_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # The fixed MES architecture also loads the pinned DeBERTa encoder.
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        import spacy
        model = load_model(sys.argv[1], json.loads(sys.argv[2]))
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
