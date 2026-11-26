# Command and option index

Generated from a build with `eval`, `discourse`, and `extractor-readability`.
Use `anno help <command>` for values and examples in your installation.
Every route accepts `--config`, `--color`, `-h, --help` and `-V, --version`.
Parent options appear on the parent row. Put them before the nested action.
See the [quickstart](QUICKSTART.md) for profile applicability.

| Invocation | Local options |
| --- | --- |
| `anno [OPTIONS] [TEXT]... [COMMAND]` | — |
| `anno extract [OPTIONS] [POSITIONAL]...` | `-t, --text`, `-f, --file`, `--input-source-map`, `-m, --model`, `-l, --label`, `--types`, `--extract-types`, `--extract-relations`, `--relation-types`, `--relation-threshold`, `--relation-max-span-distance`, `--threshold`, `--expected-types`, `--format`, `--context-window`, `--include-sentence`, `--export`, `--export-graph`, `--url`, `--html-source-spans`, `--clean`, `--normalize`, `--detect-lang`, `--export-format`, `--negation`, `--quantifiers`, `-v, --verbose`, `-q, --quiet` |
| `anno joint [OPTIONS] --checkpoint <CHECKPOINT> --schema <SCHEMA>` | `--checkpoint`, `--schema`, `--text`, `--window-words`, `--overlap-words`, `--python` |
| `anno debug [OPTIONS] [TEXT]...` | `-t, --text`, `-f, --file`, `--url`, `--clean`, `--normalize`, `--detect-lang`, `--export-graph`, `-m, --model`, `--html`, `--export`, `--export-format`, `-o, --output`, `--coref`, `--link-kb`, `-q, --quiet`, `-v, --verbose` |
| `anno eval [OPTIONS] [POSITIONAL]...` | `-t, --text`, `-f, --file`, `-m, --model`, `--predictions`, `--metric`, `-g, --gold`, `--gold-file`, `-o, --output`, `--json`, `--html`, `-v, --verbose`, `-q, --quiet` |
| `anno validate [OPTIONS] <FILES>...` | — |
| `anno analyze [OPTIONS] [POSITIONAL]...` | `-t, --text`, `-f, --file` |
| `anno corpus [OPTIONS] --store <STORE> <COMMAND>` | `--store` |
| `anno corpus --store <STORE> preview [OPTIONS] <OPERATION>` | `--document` |
| `anno corpus --store <STORE> apply [OPTIONS] <OPERATION>` | `--document`, `--approval` |
| `anno corpus --store <STORE> read [OPTIONS] <DOCUMENT_ID>` | — |
| `anno corpus --store <STORE> read-many [OPTIONS] <DOCUMENT_ID>...` | `--at-head` |
| `anno dataset [OPTIONS] <COMMAND>` | — |
| `anno dataset list [OPTIONS]` | `-t, --task`, `-d, --domain`, `--loadable`, `-v, --verbose` |
| `anno dataset info [OPTIONS] --dataset <DATASET>` | `-d, --dataset` |
| `anno dataset eval [OPTIONS]` | `-d, --dataset`, `-m, --model`, `-t, --task`, `--html`, `--output`, `--max-cases`, `--min-errors`, `--min-case-errors`, `--min-gold-mentions`, `--min-gold-relations`, `--coref-oracle-mentions` |
| `anno dataset export [OPTIONS] --input <PATH> --output <DIR>` | `-i, --input`, `-o, --output`, `-f, --format`, `-m, --model`, `--overwrite`, `--include-confidence`, `--base-uri`, `--document-iri`, `-q, --quiet` |
| `anno dataset import [OPTIONS] --input <PATH> --output <PATH>` | `-i, --input`, `-o, --output`, `-f, --format`, `--source-text`, `--require-brat-manifest`, `--include-text`, `--allow-partial`, `-q, --quiet` |
| `anno dataset context [OPTIONS]` | `-i, --input`, `-t, --text`, `-m, --model`, `--window`, `--full-sentence`, `--format`, `-o, --output`, `--entity-type`, `-q, --quiet` |
| `anno dataset check [OPTIONS]` | `-i, --issues-only`, `-d, --dataset` |
| `anno dataset check-health [OPTIONS]` | `-d, --dataset`, `--all`, `--relaxed`, `-v, --verbose`, `-w, --workers`, `-t, --timeout` |
| `anno dataset download [OPTIONS] --dataset <DATASET>` | `-d, --dataset` |
| `anno dataset facets [OPTIONS]` | `--touched-report`, `--gaps` |
| `anno benchmark [OPTIONS]` | `--profile`, `--seeds`, `--split-heavy-backends`, `-t, --tasks`, `-d, --datasets`, `-b, --backends`, `-m, --max-examples`, `--seed`, `--cached-only`, `--relation-schema`, `-o, --output`, `--output-json` |
| `anno sampler [OPTIONS] <COMMAND>` | `--history-file`, `--slice`, `--perspective`, `--slice-by-dataset-facets`, `--facet-datasets`, `--strategy`, `--mode`, `--include-ml` |
| `anno sampler eligibility [OPTIONS] --task <TASK>` | `--task`, `--require-cached` |
| `anno sampler stats [OPTIONS]` | `--show-datasets`, `--top-datasets` |
| `anno sampler decisions [OPTIONS]` | `--file`, `--top`, `--by-run`, `--run-filter`, `--top-runs`, `--trend-window`, `--by-arm`, `--by-dataset` |
| `anno sampler decide [OPTIONS]` | `--k`, `--per-dataset`, `--datasets`, `--show-frontier`, `--show-candidates`, `--top-candidates` |
| `anno sampler run [OPTIONS]` | `--runs`, `--seed-base`, `--decisions-file`, `--agg-out`, `--task`, `--max-examples`, `--datasets-per-run`, `--backends-per-run`, `--per-dataset`, `--require-cached`, `--try-download-on-empty`, `--fixed-datasets`, `--fixed-backend`, `--pin-lang`, `--pin-domain`, `--pin-backend` |
| `anno sampler regress [OPTIONS]` | `--mode`, `--recent`, `--min-recent-calls`, `--top`, `--datasets`, `--include-global` |
| `anno info [OPTIONS]` | — |
| `anno models [OPTIONS] <COMMAND>` | — |
| `anno models list [OPTIONS]` | — |
| `anno models info [OPTIONS] <MODEL>` | — |
| `anno models compare [OPTIONS]` | — |
| `anno models download [OPTIONS] <MODEL>...` | `--include-relation` |
| `anno parse [OPTIONS]` | `-i, --input`, `--format` |
| `anno cross-doc [OPTIONS] [DIR]` | `-m, --model`, `-t, --threshold`, `--candidates`, `--explain-pair`, `--require-type-match`, `-f, --format`, `--import`, `--delivery-manifest`, `--stdin`, `--extensions`, `-r, --recursive`, `--min-cluster-size`, `--cross-doc-only`, `--type`, `--max-clusters`, `-o, --output`, `-v, --verbose` |
| `anno enhance [OPTIONS] <FILE>` | `--coref`, `--link-kb`, `-e, --export`, `--export-format`, `--format`, `-q, --quiet`, `--export-graph` |
| `anno pipeline [OPTIONS] [TEXT]...` | `-f, --files`, `-d, --dir`, `-m, --model`, `--coref`, `--maverick-model`, `--speaker-turns`, `--link-kb`, `--cross-doc`, `--threshold`, `--require-type-match`, `--format`, `-o, --output`, `--progress`, `-q, --quiet` |
| `anno query [OPTIONS] <FILE>`<br>`anno query <COMMAND>` | `--catalog-record`, `-t, --type`, `-e, --entity`, `--min-confidence`, `-f, --filter`, `--start-offset`, `--end-offset`, `--negated`, `--quantified`, `--untracked`, `--linked`, `--unlinked`, `--format`, `-o, --output` |
| `anno query publication [OPTIONS] <CURRENT>` | `--since`, `--catalog-record` |
| `anno query edition [OPTIONS] --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> <COMMAND>` | `--collection`, `--source`, `--audience` |
| `anno query edition --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> manifest [OPTIONS] <EDITION> [SNAPSHOTS]...` | — |
| `anno query edition --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> stage [OPTIONS] <DATABASE> <RECEIPT>` | — |
| `anno query edition --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> publish [OPTIONS] <DATABASE> <MANIFEST>` | — |
| `anno query edition --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> read [OPTIONS] <DATABASE> [EDITION]` | `--min-edition` |
| `anno query edition --collection <COLLECTION> --source <SOURCE> --audience <AUDIENCE> evict [OPTIONS] <DATABASE> <CHECKPOINT>` | — |
| `anno query memberships [OPTIONS] <CURRENT>` | `--since`, `--neighbors`, `--explain-pair` |
| `anno query predecessors [OPTIONS] --prior <PRIOR> --doc-id <DOC_ID> --track-id <TRACK_ID> <CURRENT>` | `--prior`, `--doc-id`, `--track-id`, `--limit`, `--minimum-score` |
| `anno query relations [OPTIONS] <INPUT>` | — |
| `anno query source-span [OPTIONS] <INPUT> <START> <END>` | `--original` |
| `anno compare [OPTIONS] <FILE1> [FILE2]` | `--models`, `--model-list`, `--format`, `--confidence-epsilon`, `--changes-only`, `-o, --output` |
| `anno cache [OPTIONS] <COMMAND>` | — |
| `anno cache list [OPTIONS]` | — |
| `anno cache clear [OPTIONS]` | `--confirm` |
| `anno cache stats [OPTIONS]` | — |
| `anno cache invalidate [OPTIONS]` | `--model`, `--file` |
| `anno cache sync-s3 [OPTIONS]` | `--bucket`, `--datasets`, `--limit`, `--dry-run` |
| `anno history [OPTIONS] <COMMAND>` | `--history-file` |
| `anno history stats [OPTIONS]` | — |
| `anno history recent [OPTIONS] <BACKEND>` | `-l, --limit` |
| `anno history best [OPTIONS] <BACKEND>` | `-d, --dataset`, `-l, --limit` |
| `anno history range [OPTIONS] <START> <END>` | `-b, --backend` |
| `anno history compare [OPTIONS] <BACKEND1> <BACKEND2>` | `-d, --dataset` |
| `anno history backends [OPTIONS]` | — |
| `anno history datasets [OPTIONS]` | — |
| `anno history rebuild [OPTIONS]` | — |
| `anno config [OPTIONS] <COMMAND>` | — |
| `anno config save [OPTIONS] <NAME>` | `--model`, `--format`, `--coref`, `--link-kb`, `--threshold` |
| `anno config list [OPTIONS]` | — |
| `anno config show [OPTIONS] <NAME>` | — |
| `anno config delete [OPTIONS] <NAME>` | — |
| `anno batch [OPTIONS]` | `-d, --dir`, `--stdin`, `-m, --model`, `--coref`, `--link-kb`, `-p, --parallel`, `--batch-size`, `--progress`, `--cache`, `-o, --output`, `--format`, `-q, --quiet` |
| `anno privacy [OPTIONS]` | `-i, --input`, `-t, --text`, `-a, --action`, `-o, --output`, `-m, --model`, `--export-map`, `--types`, `-q, --quiet` |
| `anno watch [OPTIONS] --directory <DIR>` | `-d, --directory`, `-o, --output`, `--extensions`, `-m, --model`, `--format`, `--interval`, `--initial`, `-q, --quiet`, `--max-files` |
| `anno domain [OPTIONS]` | `-i, --input`, `-t, --text`, `-m, --model`, `--format`, `-q, --quiet` |
| `anno explain [OPTIONS] [POSITIONAL]...` | `-e, --entity-id`, `-t, --text`, `-s, --span`, `-m, --model`, `--show-all` |
| `anno singleton [OPTIONS]` | `-i, --input`, `-t, --text`, `-m, --model`, `--format`, `-v, --verbose`, `-q, --quiet` |
| `anno export [OPTIONS] --input <PATH> --output <DIR>` | `-i, --input`, `-o, --output`, `-f, --format`, `-m, --model`, `--overwrite`, `--include-confidence`, `--base-uri`, `--document-iri`, `-q, --quiet` |
| `anno completions [OPTIONS] <SHELL>` | — |
