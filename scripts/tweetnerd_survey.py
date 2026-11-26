"""Count public TweetNERD annotations without hydrating tweets.

Pass one or more released TSV files to print counts over their union. Academic
and OOD are named evaluation splits. The publisher describes part_0 through
part_11 as the remaining records in no particular order. Source and schema:
https://doi.org/10.5281/zenodo.6617192
"""

import argparse
import csv
import itertools
import json
import re
from collections import Counter
from pathlib import Path

HEADER = ["tweet_id", "phrase", "start", "end", "entityId", "score"]
NO_MENTION = ["NO_PHRASE", "-1", "-1", "NO_ENTITY", "-1"]


def count_annotations(paths):
    tweets, spans, unique_rows = set(), set(), set()
    counts = Counter()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, delimiter="\t")
            first = next(reader, None)
            if first is None:
                raise ValueError("Empty TweetNERD source")
            rows = reader if first == HEADER else itertools.chain([first], reader)
            for row in rows:
                if (
                    len(row) != len(HEADER)
                    or not row[0].isascii()
                    or not row[0].isdigit()
                ):
                    raise ValueError("Invalid TweetNERD row or header")
                tweet, phrase, start, end, entity, score = row
                tweets.add(tweet)
                counts["rows"] += 1
                unique_rows.add(tuple(row))
                if entity == "NO_ENTITY":
                    if row[1:] != NO_MENTION:
                        raise ValueError("Inconsistent TweetNERD no-entity sentinel")
                    counts["no_entity_rows"] += 1
                    continue
                start, end = int(start), int(end)
                if (
                    start < 0
                    or end <= start
                    or int(score) < 1
                    or end - start != len(phrase.encode("utf-16-be")) // 2
                ):
                    raise ValueError(
                        "Invalid TweetNERD mention, UTF-16 offsets or score"
                    )
                counts["mention_rows"] += 1
                spans.add((tweet, start, end))
                if re.fullmatch(r"Q[0-9]+", entity):
                    counts["wikidata_link_rows"] += 1
                elif entity in {"NOT_FOUND", "AMBIGUOUS"}:
                    counts[entity.lower() + "_rows"] += 1
                else:
                    # Preserve a released malformed identifier as an observed span.
                    counts["other_link_value_rows"] += 1
    counts.update(
        unique_tweets=len(tweets),
        unique_spans=len(spans),
        duplicate_rows=counts["rows"] - len(unique_rows),
    )
    return counts


def describe(path):
    counts = count_annotations([path])
    return dict(
        records=counts["rows"],
        unit="released annotation rows including no-entity placeholders",
        reason=(
            f"{counts['unique_tweets']} distinct tweet IDs; {counts['unique_spans']} "
            f"distinct UTF-16 mention spans; {counts['no_entity_rows']} no-entity rows; "
            f"{counts['duplicate_rows']} duplicate rows. Complete tweet text is absent."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    print(json.dumps(count_annotations(parser.parse_args().files), sort_keys=True))
