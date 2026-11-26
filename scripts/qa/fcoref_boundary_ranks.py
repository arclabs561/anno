"""Score FCoref pre-pruning ranks against the 256 frozen GAP control articles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

from backend_residuals.private_ledger import external_path
from coref_gap import DATA_SHA, check_file
from native_gap_possessives import VALIDATION_SHA, read_rows


ORDERED_IDS = 'a76cd45d822c8d7abd003984f648b164d4fda1859427a238daa4eb4c631982da'
STOCK_PARTITIONS = '758cd70d7f1b2e79da4bc0d858ffd14094f8efb9802ee757d98f1d3ba374ed4e'
TOKENIZER = '5ec67f22fb423648c32c22819353703e5d3dddd3a553b11989f37702fb910aa1'


def cohort(ids, test, validation):
    check_file(test, DATA_SHA)
    check_file(validation, VALIDATION_SHA)
    ordered = ids.read_text().split()
    if len(set(ordered)) != 257 or hashlib.sha256('\n'.join(ordered).encode()).hexdigest() != ORDERED_IDS:
        raise ValueError('Expected the 257 ordered IDs including the issue case')
    rows = {r['ID']: r for r in read_rows(validation) + read_rows(test)}
    return [rows[k] for k in ordered[1:]]  # omit the case used to form the original hypothesis


def ranks(path):
    with path.open() as source:
        for line in source:
            if line.startswith('FCOREF_RANKS '):
                yield json.loads(line.removeprefix('FCOREF_RANKS '))


def prediction(path):
    with path.open() as source:
        for line in source:
            yield json.loads(line)


def score(rows, old, new, traces, tokenizer):
    splits = {}
    admitted = linked = absent = 0
    margins, cuts, positions = [], [], []
    digest = hashlib.sha256()
    for row, stock, double, (k, candidates) in zip(rows, old, new, traces, strict=True):
        if any(r['id'] != row['ID'] or r['producer'] != 'fcoref' for r in (stock, double)):
            raise ValueError('Wrong document or backend in predictions')
        norm = sorted(sorted(c) for c in stock['clusters'])
        digest.update(json.dumps([row['ID'], norm], separators=(',', ':')).encode() + b'\n')
        text = row['Text']
        offsets = tokenizer.encode(text).offsets
        if k != (len(offsets) + 3) // 4 or len(candidates) != sum(min(30, len(offsets)-i) for i in range(len(offsets))):
            raise ValueError('Incorrect stock budget or incomplete rank table')
        # The first token-pair mapped to a source span is its best by the native stable ordering.
        span_ranks = {}
        for rank, (i, j, logit) in enumerate(candidates, 1):
            if rank > 1 and logit > candidates[rank - 2][2]:
                raise ValueError('Candidate rows are out of order')
            a, b = offsets[i][0], offsets[j][1]
            surface = text[a:b]
            if surface.strip():
                left = a + len(surface) - len(surface.lstrip())
                span_ranks.setdefault((left, left + len(surface.strip())), (rank, logit))
        def marked(field):
            start = int(row[field + '-offset'])
            return start, start + len(row[field])
        pronoun = marked('Pronoun')
        groups = []
        for run in (stock, double):
            clusters = [set(map(tuple, group)) for group in run['clusters']]
            groups.append(next((group for group in clusters if pronoun in group), set()))
        counts = splits.setdefault(row['ID'].split('-')[0], Counter())
        for field in ('A', 'B'):
            name = marked(field)
            positive = row[field + '-coref'] == 'TRUE'
            for arm, cluster in zip(('stock', 'double'), groups, strict=True):
                correct = positive == (name in cluster)
                counts[f'{arm}_' + (('tp' if positive else 'tn') if correct else ('fn' if positive else 'fp'))] += 1
            counts['changed_decisions'] += (name in groups[0]) != (name in groups[1])
            if not positive:
                continue
            rank, logit = span_ranks[name]
            if rank <= k:
                continue
            absent += 1
            rivals = [s for s in groups[0] if s[0] <= name[0] < name[1] <= s[1]
                      or name[0] <= s[0] < s[1] <= name[1]]
            if rivals:
                margins.append(max(span_ranks[s][1] for s in rivals) - logit)
                cuts.append(candidates[k-1][2] - logit)
                positions.append(rank - k)
                admitted += rank <= 2*k
                linked += name in groups[1]
    if digest.hexdigest() != STOCK_PARTITIONS:
        raise ValueError('Stock clusters differ from the original native reproduction')
    def summary(values):
        return dict(min=min(values), median=median(values), max=max(values))
    return dict(splits=splits, missing_exact_name=absent, with_linked_boundary_rival=len(margins),
                admitted_by_double=admitted, linked_by_double=linked, rival_logit_gap=summary(margins),
                cutoff_logit_gap=summary(cuts), ranks_below_cut=summary(positions))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('ids', 'test', 'validation'):
        cli.add_argument(name, type=Path)
    cli.add_argument('--prepare-input', type=Path, help='Create source-bearing JSONL in a new external file')
    for name in ('stock', 'double', 'trace', 'tokenizer'):
        cli.add_argument('--' + name, type=Path)
    args = cli.parse_args()
    rows = cohort(args.ids, args.test, args.validation)
    if args.prepare_input:
        with external_path(args.prepare_input).open('x', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(dict(id=row['ID'], text=row['Text'])) + '\n')
        return
    if any(path is None for path in (args.stock, args.double, args.trace, args.tokenizer)):
        cli.error('Provide stock, double, trace and tokenizer to score')
    from tokenizers import Tokenizer
    check_file(args.tokenizer, TOKENIZER)
    print(json.dumps(score(rows, prediction(args.stock), prediction(args.double),
                           ranks(args.trace), Tokenizer.from_file(str(args.tokenizer))), indent=2))


if __name__ == '__main__':
    main()
