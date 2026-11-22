# Live source captures

Run the following command from the repository root to download pages and papers for local extraction checks.

```sh
scripts/fetch_testdata.sh
```

Only this README is tracked here. The helper writes ignored HTML captures for EU, UN, OpenAI and arXiv pages and PDF captures for the two arXiv paper endpoints. Output filenames include the host and URL path, for example `arxiv_org_pdf_2309_14084.pdf`. It reports the saved byte counts.

The helper follows redirects, checks the final HTTP status, and rejects bodies shorter than 100 bytes. PDF captures must also begin with a PDF header. If a download fails validation, any previous capture at that path survives. Other URLs are still attempted, and the command exits with a failure status if any capture failed.

These sources can change or serve access notices. The captures have no gold annotations. Use the [pinned public-source excerpts](../../crates/anno-cli/tests/fixtures/public_sources/README.md) for reproducible span checks.
