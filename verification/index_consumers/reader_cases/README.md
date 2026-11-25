# Answer questions about corrected credits

Run the Rust reader and the independent SQLite reader against the same published inputs and answer key from the repository root.

```sh
python3 verification/index_consumers/reader_cases/check.py --bounded
```

The checkout supplies the `reader_answer` Rust example and `anno query publication`. The command builds the binaries by default. Use `--anno PATH --rust-reader PATH` to check a specific pair. The bounded replay uses original synthetic histories. The [fixture guide](fixtures/README.md) records the sources and reuse terms for the hand-authored cases.

| Reader question | Operation | Evidence used |
|---|---|---|
| What does this source edition say? | `printed` | Visible source, edition and character range. The Rust example opens a citation against its text edition. The SQLite reader checks the CLI's returned document, revision, digest and span. |
| Which record does this reference point to here? | `reference` | An explicit reference and any visible one-hop redirect at the selected content revision. |
| Who received this credit for this work? | `credit` | Active assertions for the requested role. Select an issuer when the question asks what a particular publisher says. |
| Why is this person in the result? | `evidence` | IDs of the visible assertions that support this role and qualified target. Independent support can survive a withdrawal. |
| Can I cite what this issuer actually published? | `cited_credit` | One receipt per visible issuer and local assertion ID, with its currently permitted excerpt. Staff can reopen its exact source edition and range. |

The [credit citation client](citation_client.py) uses the targets returned by its first query to ask two independent readers for the supporting publication passages. Both examples accept a caller-supplied question with `--ask`. The [fixture guide](fixtures/README.md) gives the wire format and invocation. Use the [finite support model](formal/README.md) to replay independent issuer withdrawals and present policy on historical editions.

Source and issuer are separate selectors. One set of published minutes can report two issuers assigning a role to different people. Each work has exact content revisions. A trusted application supplies the viewer and applicable policy revision for the read, including when opening a historical content edition. These local examples receive that decision as an input. A public service must bind it to its authenticated request. Empty public results can represent absence, unresolved evidence, or withheld support. The raw input and intermediate databases contain the full evidence.

The [snippet reader](snippets.py) returns just the assertion ID, issuer and approved excerpt to a public viewer. Staff can inspect its full source locator. Changing a neighboring withheld sentence changes the source digest, so even that digest stays in the privileged view. Replacing a credit under the same assertion ID requires a new excerpt approval for the new revision.

## Propose and judge another history

The bounded runner checks every three-event sequence of grants, withdrawals, hides and reinstatements in its two-publisher model. At each prefix, it asks both readers about every available content edition for public and staff viewers. The independent answer key keeps two facts about support and one visibility switch. The fixed corpus also covers shared bylines, literal Unicode ranges, independent references after a redirect is removed, and input already conflated when it arrived. These checks assume one work per case and at most one applicable redirect hop.

An adversarial agent or a developer can select a longer witness and replay it with the same offline checker. Save exploratory files in a local round directory.

```sh
case_dir=$(mktemp -d)
suite=verification/index_consumers/reader_cases
python3 "$suite/bounded.py" grant-b hide-a drop-b > "$case_dir/cases.json"
python3 "$suite/bounded.py" --key grant-b hide-a drop-b > "$case_dir/key.json"
python3 "$suite/check.py" --cases "$case_dir/cases.json" --answer-key "$case_dir/key.json"
```

Give a proposal agent the event vocabulary or fixture inputs. Keep the expected answers with the independent scorer. Minimize a failing proposal by removing events and rerunning the failing query. Promote it to shared replay with an explanation, source rights, and the expected result. For a new predicate outside the bounded model, write a separate case and answer key using the fixture format.

The adjacent [edition replay](../check_editions.py) tests duplicated and reordered deliveries, missing checkpoints, and a publisher that explicitly selects an older member again. The [credit search replay](../credits/check_search.py) checks exact hits, counts, bylines and autocomplete and includes an intentionally faulty postings reader. Neither a collection edition number nor a catalog target carries a viewer permission by itself.
