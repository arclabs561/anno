"""Compare surface and predicted dependency heads on whole CorefUD development documents."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from corefud_retrieval import CHECKSUM, REV, score


MODELS = Path(__file__).with_name('udpipe-models.json')
MODEL_REVISION = '17d6bbfb7f138f9ba2cf04692f15d892becf0246'
SALT = b'anno-corefud-predicted-head-1\0'


def doc_key(doc):
    # Group on source content, before looking at any annotations.
    return hashlib.sha256(SALT + doc['text'].encode()).digest()


def component(file_name):
    name, marker, suffix = file_name.partition('-corefud-dev')
    if not marker or not name or not suffix.startswith('_') or not suffix[1:].isdigit():
        raise ValueError('Unexpected CorefUD development document identifier')
    return name


def select(shard, per_corpus):
    groups = defaultdict(set)
    for batch in shard.iter_batches(columns=['file_name', 'text'], batch_size=16):
        for doc in batch.to_pylist():
            groups[component(doc['file_name'])].add(doc_key(doc))
    return {name: set(sorted(keys)[:per_corpus]) for name, keys in groups.items()}


def panel_digest(groups):
    digest = hashlib.sha256()
    for name, keys in sorted(groups.items()):
        for key in sorted(keys):
            digest.update(name.encode() + b'\0' + key)
    return digest.hexdigest()


def dependencies(tokens, model):
    from ufal.udpipe import Model, ProcessingError, Sentence

    parents = []
    for _, sentence_tokens in groupby(tokens, key=lambda t: t['sentence_ID']):
        sentence = Sentence()
        first = len(parents)
        for token in sentence_tokens:
            sentence.addWord(token['form'])
        error = ProcessingError()
        if not model.tag(sentence, Model.DEFAULT, error) or not model.parse(sentence, Model.DEFAULT, error):
            raise RuntimeError('UDPipe failed to tag or parse a sentence')
        for word in list(sentence.words)[1:]:
            if not 0 <= word.head < len(sentence.words):
                raise ValueError('Parser returned an invalid head')
            parents.append(first + word.head - 1 if word.head else -1)
    return parents


def span_heads(tokens, parents, spans, text):
    """Accept one exact source-aligned lexical governor per supplied mention."""
    if len(parents) != len(tokens):
        raise ValueError('Parser output and source token counts differ')
    heads = {}
    for left, right, start, end in spans:
        if not all(isinstance(coord, int) for coord in (left, right, start, end)):
            continue
        if not (0 <= left < right <= len(text) and 0 <= start <= end < len(tokens)):
            continue
        if (tokens[start]['char_onset'], tokens[end]['char_offset']) != (left, right):
            continue
        if tokens[start]['sentence_ID'] != tokens[end]['sentence_ID']:
            continue
        # A unique token whose governor is outside this span is its root.
        roots = [i for i in range(start, end + 1) if not start <= parents[i] <= end]
        if len(roots) == 1:
            token = tokens[roots[0]]
            a, b = token['char_onset'], token['char_offset']
            if isinstance(a, int) and isinstance(b, int) and left <= a < b <= right and text[a:b] == token['form']:
                heads[left, right] = a, b
    return heads


def load_model(spec, directory):
    from ufal.udpipe import Model

    path = directory / spec['filename']
    with path.open('rb') as source:
        digest = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
        for chunk in iter(lambda: source.read(1 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != spec['git_blob']:
        raise ValueError(f'{path.name}: pinned model verification failed')
    model = Model.load(str(path))
    if model is None:
        raise ValueError(f'{path.name}: UDPipe could not load this model')
    return model


def intervals(rows):
    """Resample whole documents within each component to preserve panel composition."""
    groups = defaultdict(list)
    for corpus, counts in rows:
        groups[corpus].append((counts['all.queries'], counts['gained'] - counts['lost']))
    rng = random.Random(20261126)
    deltas = []
    for _ in range(2000):
        queries = gains = 0
        for group in groups.values():
            for n, delta in rng.choices(group, k=len(group)):
                queries += n
                gains += delta
        if queries:
            deltas.append(gains / queries)
    if not deltas:
        return None
    bounds = statistics.quantiles(deltas, n=40, method='inclusive')
    return [round(bounds[0], 6), round(bounds[-1], 6)]


def native_counts(doc, parents, clusters):
    # Gold IDs only label the scorer. All predicted spans, assignments, and
    # parser inputs originate in text-only model output or supplied source tokens.
    tokens, text = doc['tokens'], doc['text']
    starts, ends = defaultdict(list), defaultdict(list)
    for i, token in enumerate(tokens):
        starts[token['char_onset']].append(i)
        ends[token['char_offset']].append(i)
    reference = defaultdict(list)
    for m in doc['mentions']:
        reference[m['onset'], m['offset']].append(m['COREF'])
    intersection = Counter()
    predicted_pairs = 0
    supplied = []
    spans = []
    seen = set()
    for cluster_id, cluster in enumerate(clusters):
        retained = 0
        for a, b in cluster:
            if not (0 <= a < b <= len(text)) or (a, b) in seen:
                raise ValueError('Invalid or duplicate native output span')
            seen.add((a, b))
            if not any(c.isalnum() for c in text[a:b]):
                continue
            labels = reference.get((a, b), ())
            target = labels[0] if len(labels) == 1 else ('unmatched', a, b)
            intersection[cluster_id, target] += 1
            retained += 1
            i = starts[a][0] if len(starts[a]) == 1 else -1
            j = ends[b][0] if len(ends[b]) == 1 else -1
            spans.append((a, b, i, j))
            supplied.append(dict(onset=a, offset=b, start_token=i, end_token=j, COREF=target))
        predicted_pairs += retained * (retained - 1) // 2
    heads = span_heads(tokens, parents, spans, text) if parents is not None else {}
    candidate = score(dict(text=text, mentions=supplied), predicted_heads=heads)
    counts = Counter(
        native_output_mentions=len(supplied),
        native_exact_mentions=sum(len(reference.get((a, b), ())) == 1 for a, b, _, _ in spans),
        native_predicted_pairs=predicted_pairs,
        native_correct_pairs=sum(n * (n - 1) // 2 for n in intersection.values()),
    )
    for policy in ('surface', 'head_backoff'):
        for metric in ('correct_pairs', 'predicted_pairs'):
            counts[f'native_spans.{policy}.{metric}'] = candidate[f'{policy}.{metric}']
    return counts


def run(cache, model_dir, per_corpus, native_runner=None):
    import pyarrow
    import pyarrow.parquet as pq
    from importlib.metadata import version

    with cache.open('rb') as data:
        if hashlib.file_digest(data, 'sha256').hexdigest() != CHECKSUM:
            raise ValueError('Expected the pinned CorefUD 1.4 development shard')
    shard = pq.ParquetFile(cache)
    selected = select(shard, per_corpus)
    specs = json.loads(MODELS.read_text())
    totals = Counter()
    languages = defaultdict(Counter)
    observed = defaultdict(set)
    rows = []
    parse_ms = []
    unsupported = Counter()
    last_language = model = None
    for batch in shard.iter_batches(columns=['file_name', 'language', 'text', 'tokens', 'mentions'], batch_size=4):
        for doc in batch.to_pylist():
            corpus, language = component(doc['file_name']), doc['language']
            key = doc_key(doc)
            if key not in selected[corpus] or key in observed[corpus]:
                continue
            observed[corpus].add(key)
            if language not in specs:
                unsupported[language] += 1
                continue
            if language != last_language:
                print(f'Parsing {language} with {specs[language]["filename"]}', file=sys.stderr, flush=True)
                model = load_model(specs[language], model_dir)
                last_language = language
            tokens = doc['tokens']
            spans = [(m['onset'], m['offset'], m['start_token'], m['end_token']) for m in doc['mentions']]
            started = time.perf_counter()
            try:
                parents = dependencies(tokens, model)
            except RuntimeError:
                heads = {}
                parents = None
                failures = 1
            else:
                heads = span_heads(tokens, parents, spans, doc['text'])
                failures = 0
            parse_ms.append((time.perf_counter() - started) * 1000)
            counts = score(doc, predicted_heads=heads)
            if native_runner is not None:
                request = dict(id=key.hex(), text=doc['text'])
                response = subprocess.run([str(native_runner), 'mention-ranking'], check=True,
                                          input=json.dumps(request) + '\n', text=True, capture_output=True)
                prediction = json.loads(response.stdout)
                if prediction['id'] != key.hex() or prediction['producer'] != 'mention-ranking':
                    raise ValueError('Unexpected native producer or document identity')
                counts.update(native_counts(doc, parents, prediction['clusters']))
            counts['parser_failures'] += failures
            counts['tokens'] += len(tokens)
            languages[language].update(counts)
            totals.update(counts)
            rows.append((corpus, counts))
    if observed != selected:
        raise ValueError('Selected panel did not replay exactly')
    native_hash = None
    if native_runner:
        with native_runner.open('rb') as binary:
            native_hash = hashlib.file_digest(binary, 'sha256').hexdigest()
    return dict(revision=REV, sha256=CHECKSUM, model_revision=MODEL_REVISION, native_runner_sha256=native_hash,
                pyarrow=pyarrow.__version__, udpipe=version('ufal.udpipe'),
                documents_per_component_cap=per_corpus, components=len(selected), panel_sha256=panel_digest(selected),
                unsupported_language_documents=unsupported, total=totals, languages=languages,
                paired_retrieval_delta_ci95=intervals(rows),
                parse_and_alignment_seconds=round(sum(parse_ms) / 1000, 3),
                parse_and_alignment_p95_ms=round(statistics.quantiles(parse_ms, n=20, method='inclusive')[-1], 2)
                if len(parse_ms) >= 2 else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('result', type=Path)
    parser.add_argument('--native-runner', type=Path, help='Optional Anno coref_gap_panel binary')
    parser.add_argument('--per-corpus', type=int, default=4)
    args = parser.parse_args()
    if args.per_corpus < 1:
        parser.error('--per-corpus must be positive')
    args.result.write_text(json.dumps(run(args.shard, args.model_dir, args.per_corpus, args.native_runner),
                                      indent=2, sort_keys=True) + '\n')
