---
name: anno-cli
description: Use the anno command-line interface for extraction, redaction, coreference, corpus queries, batch processing, model management, evaluation, and other CLI operations.
---

# Anno CLI

Start with the documentation for the Anno revision you are running. In a checkout, open `docs/README.md`. With an installed binary, check `anno --version` and use the corresponding release docs when available. The [project documentation index](https://github.com/arclabs561/anno/blob/main/docs/README.md) links the current guides.

- For installation and worked CLI examples, read `docs/QUICKSTART.md`. To reuse extraction settings, run `anno config ls` and `anno config show NAME`, then follow [Reuse a CLI workflow profile](../../../docs/QUICKSTART.md#reuse-a-cli-workflow-profile) for command-line precedence and relation defaults.
- For model selection, read `docs/BACKENDS.md`. Choose by the input and output required (typed mentions, coreference clusters or directed relations), then check which backend actually ran.
- For interpreting results, saving annotations or highlighting spans in the source, read `docs/CONTRACT.md`.
- For constituency trees, read `docs/CONSTITUENCY.md`. It covers producing parser JSON and using `anno parse` to validate or render it.
- For annotation interchange, source revision checks and tracing a result into HTML, use the worked examples in `docs/EXTRACTION_WORKFLOWS.md`.
- For scoring a result or reproducing a regression, read `docs/QA.md`. The bundled `anno-qa` skill covers multi-step investigations.

For any command, including operations without a worked example, use `anno --help` to see what the installed build exposes. Follow the relevant command path with `--help` at each level to resolve its inputs, output formats and options. Feature-enabled builds may expose additional commands.
