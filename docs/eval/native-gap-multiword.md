# Joining two native proper-name proposals

## Frozen comparison

The native GAP trace found 1,175 labeled pairs where the exact multiword name was missing and every overlapping proposal lay inside it. The English fallback detector checks one whitespace-delimited word at a time. This experiment adds one candidate at the proposal stage and runs the full ranker again. [Prior stage audit](native-gap-endpoint-stages.md)

For English input with NER disabled, take the fallback's existing proper-name tokens in source order. A token qualifies when its first character is uppercase and each remaining character is a lowercase letter. Join a maximal run of exactly two qualifying proposals whose source spans are separated by one ASCII space. The new span takes the second token's head and both original proposals stay available for the stock overlap, feature and scoring routines. Sentence-initial exclusions still apply to the constituent proposals. [Candidate and project-authored control](../../scripts/qa/fixtures/native-gap-multiword.patch)

Freeze the rule before scoring any candidate output. Run development first, then test and validation, with the same rule throughout. Count exact proposed, retained and returned gold endpoint pairs, correct and wrong links gained and lost, and additions and removals of emitted spans outside GAP's three marked coordinates. A split passes only if the candidate produces a net gain in correct links and introduces zero newly wrong decisions. Previously wrong decisions removed cannot offset newly wrong ones. All three official splits have been studied previously. Adopting a default would also require evidence on an independent source with annotations relevant to its proposed use.
