"""Project the first three MultiNERD BIO label pairs to Unicode character spans."""


def text_and_gold(rows):
    offset = 0
    sentences, gold = [], []
    for row in rows:
        sentences.append(" ".join(row["tokens"]))
        prev = 0
        for token, tag in zip(row["tokens"], row["ner_tags"], strict=True):
            end = offset + len(token)
            if 1 <= tag <= 6:
                label = (tag - 1) // 2
                if tag % 2:
                    gold.append([offset, end, label])
                elif prev in (tag - 1, tag):
                    gold[-1][1] = end
                else:
                    raise ValueError("Orphan I tag in source data")
            prev = tag
            offset = end + 1
    return "\n".join(sentences), gold
