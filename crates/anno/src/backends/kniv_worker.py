"""Ground the pinned Kniv v7 runtime's window-local outputs in the caller's text."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# The release ships Python code beside its weights. Load only its pinned package,
# without adding the model directory to the interpreter's module search path.
SOURCES = {
    "__init__.py": "e3a9d154864587ac205003f755d3702189f2b41ff20af53f0064e83b9415ae41",
    "entities.py": "90034b74435781b1c054b7b1b96778aaede188ec9c8ac209593b10c9b9437e1e",
    "runtime.py": "56c0e9238ed8c172a4cad41da5d36cb79a7daee21a663e20d6721e0d63e1d7f5",
    "windows.py": "c2529772d452d68f8b20818bfc26b15ff67104079ed40568197853c46a1e3ea1",
}


FILES = {
    "calibration.json": "8f9a9c8ce9a1e0537240266c4f3f2877c4e03a72e28be6cc83baf8d4176b2f8d",
    "label_bank.json": "68b8e93dbcf2e1f88d8f2e1ff79220e3dd654c6d4667250fa54943205658865e",
    "tokenizer.json": "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    "onnx/onnx_meta.json": "3720634e0c384495c06e52c15ac017a487ba74795a1c3091d59f525e7e097cb3",
    "onnx/attr-int8.onnx": "f36cfcf6efbcc64cfa5c729c46ac3f6cf889d1865e245d7912ee19317a965006",
    "onnx/coref_spans-int8.onnx": "2cf6aaeb3073e55a581060dcff61ba17f0ffb14ba5f394e03dfbed02b3b20ba6",
    "onnx/dep_rel-int8.onnx": "85c11432e2ae3ef97d755833feb6f7fc3ddc7e345a21bff9e9c338e99636f1da",
    "onnx/marker-int8.onnx": "35b2b79266dbd8b54b89046ced99d9e53d8725f436cb77d10d0429825cd2b593",
    "onnx/plain-int8.onnx": "ff7d821384bfb9a58f86cbab80ed4f34442e349598a1e0f70c47052975a1a6fe",
    "onnx/span-int8.onnx": "eab6f21c60d07232a98b18de78aead4e9514a33bcc0ce026ee1987c2935f32c7",
    "onnx/srl_roles-int8.onnx": "d78a3296bdd1770df53fd1734b1a81d127ce491bd6887a73d0fda5b7d637ce3f",
    "onnx/label.onnx": "8c0ad13b92e86363e54246f42e4f123bb26a249bf19af3cc8bf3255b8df42b32",
}


def verify(path, digest):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != digest:
            raise ValueError(f"Kniv release digest mismatch for {path.name}")


def load(root):
    import onnxruntime as ort
    package = root / "kniv_v7"
    # Use a clean import directory. A pre-existing bytecode cache or an extension
    # module next to a downloaded source file must not override the verified code.
    with TemporaryDirectory() as clean:
        isolated = Path(clean)
        for name, digest in SOURCES.items():
            code = (package / name).read_bytes()
            if hashlib.sha256(code).hexdigest() != digest:
                raise ValueError(f"Kniv release digest mismatch for {name}")
            (isolated / name).write_bytes(code)
        spec = importlib.util.spec_from_file_location("kniv_v7", isolated / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        from kniv_v7 import runtime
    for name, digest in FILES.items():
        if name.endswith(".json"):
            verify(root / name, digest)

    class Backend:
        def __init__(self):
            self.sessions = {}
            self.ner = None

        def run(self, name, **inputs):
            if name not in self.sessions:
                graph = root / "onnx" / ("label.onnx" if name == "label" else f"{name}-int8.onnx")
                verify(graph, FILES["onnx/" + graph.name])
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = opts.inter_op_num_threads = 1
                self.sessions[name] = ort.InferenceSession(str(graph), opts, providers=["CPUExecutionProvider"])
            session = self.sessions[name]
            outputs = session.run(None, {i.name: inputs[i.name] for i in session.get_inputs()})
            if name == "plain":
                self.ner = outputs[model.meta["graphs"]["plain"]["outputs"].index("ner")][0]
            return outputs

    model = runtime.Kniv(root, backend=Backend())
    if not (root / "onnx/plain-int8.onnx").is_file():
        raise ValueError("missing onnx/plain-int8.onnx")
    return model


def sentence_words(text):
    from kniv_v7.windows import split_sentences, tokenize
    cursor = 0
    for paragraph in text.split("\n\n"):
        if paragraph.strip():
            local_end = 0
            for sentence in split_sentences(paragraph):
                start = paragraph.index(sentence, local_end)
                # Upstream splitting may consume a closing quote along with whitespace.
                # Preserve those source tokens in their own short segment.
                for part, offset in [(paragraph[local_end:start], local_end), (sentence, start)]:
                    tokens = tokenize(part)
                    if tokens:
                        yield aligned(part, cursor + offset, tokens)
                local_end = start + len(sentence)
            tail = paragraph[local_end:]
            if tail.strip():
                yield aligned(tail, cursor + local_end, tokenize(tail))
        cursor += len(paragraph) + 2


def aligned(text, base, tokens):
    end = 0
    words = []
    for token in tokens:
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith(token, end):
            raise ValueError("Kniv tokenizer disagrees with source")
        words.append((token, base + end, base + end + len(token)))
        end += len(token)
    if text[end:].strip():
        raise ValueError("Kniv tokenizer left unanalyzed source text")
    return words


def windows(text, model, with_relations):
    """Keep upstream sentence packing while covering long sentences and subword overflow."""
    words, sentences = [], []
    for sentence in sentence_words(text):
        for start in range(0, len(sentence), 512):
            part = sentence[start:start + 512]
            if words and len(words) + len(part) > 512:
                yield from fit(words, sentences, model, with_relations)
                words, sentences = [], []
            sentences.append((len(words), len(words) + len(part)))
            words.extend(part)
    if words:
        yield from fit(words, sentences, model, with_relations)


def fit(words, sentences, model, with_relations):
    tokens = [w[0] for w in words]
    ids, _ = model.tok.encode(" ".join(tokens))
    fits = len(ids) <= model.meta["max_subwords"]
    if with_relations and fits:
        # Kniv inserts at most one opening and one closing marker per word.
        # Its relation encoder truncates independently of the plain encoder.
        from kniv_v7.runtime import MARKER, MARKER_MAX_SUBWORDS
        marked, _ = model.tok.encode(" ".join(f"{MARKER} {token} {MARKER}" for token in tokens))
        fits = len(marked) <= MARKER_MAX_SUBWORDS
    if fits:
        yield words, {"tokens": tokens, "sentence_spans": sentences}
        return
    if len(words) == 1:
        raise ValueError(f"word at character {words[0][1]} exceeds encoder capacity")
    # Preserve sentence boundaries when possible.
    cut = min((right for _, right in sentences[:-1]), default=len(words)//2,
              key=lambda pos: abs(len(words)//2 - pos))
    for lo, hi in ((0, cut), (cut, len(words))):
        clipped = [(max(lo, s)-lo, min(hi, e)-lo) for s, e in sentences if s < hi and e > lo]
        yield from fit(words[lo:hi], clipped, model, with_relations)


def analyze(model, text, relation_types, entity_types):
    import numpy as np
    from kniv_v7.runtime import log_softmax
    labels = model.labels["ner"]
    relations = [{"name": r} for r in relation_types]
    types = [{"name": t} for t in entity_types]
    results = []
    for words, window in windows(text, model, bool(relations)):
        result = model.window(window, relations, types, linguistic=True, top=model.top)
        if result["analysed_words"] != len(words):
            raise ValueError("Kniv truncated a fitted window")
        logp = log_softmax(model.backend.ner)
        def span(start, end):
            if not 0 <= start < end <= len(words):
                raise ValueError("invalid Kniv mention")
            return [words[start][1], words[end-1][2]]
        ner = []
        for label, start, end in result["ner"]:
            selected = [logp[i, labels.index(("B-" if i == start else "I-") + label)]
                        for i in range(start, end)]
            ner.append({"label": label, "span": span(start, end),
                        "confidence": float(np.exp(np.mean(selected)))})
        nodes = [{"label": node["type"], "mentions": [span(a,b) for a,b in node["spans"]]}
                 for node in result["entities"]]
        results.append({"span": span(0, len(words)), "ner": ner, "nodes": nodes,
                        "facts": result["facts"]})
    return {"windows": results}


if __name__ == "__main__":
    try:
        model = load(Path(sys.argv[1]))
        print(json.dumps("ready"), flush=True)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            answer = {"ok": analyze(model, req["text"], req["relation_types"], req["entity_types"])}
            print(json.dumps(answer, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
