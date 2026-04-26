# Location (multimodal philosophy)

`anno`’s extraction contract is **text-first**: most backends produce **character-offset spans**
over a UTF-8 string.

`anno::core` also includes a broader `Location` enum (text spans, bboxes, temporal intervals, etc.)
to support future multi-modal pipelines. Today:

- The **stable, widely-used** localization unit is `Span::Text { start, end }` (character offsets).
- Non-text `Location` variants are best treated as **experimental substrate**:
  they are useful for thinking and serialization, but most `anno` backends won’t produce them.

## Guidance

- If you’re doing regular NER/coref: use `Span`/`Entity`/`Signal` with `Location::Text`.
- If you’re doing OCR/layout work: keep the *layout model* upstream (or downstream) and only join
  it to `anno` extractions at the boundary.
- If you need a knowledge-graph substrate: export into `lattix` (triples/graphs/algorithms live
  there).

