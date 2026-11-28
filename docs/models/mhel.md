# Historical Wikidata entity linking

MHEL-LLaMo takes identified mention occurrences and ranks Wikidata candidates. The release pairs WikiBELA retrieval with an instruction-tuned chat model. Use the retriever and reranker Python environments separately, following the [pinned authors’ installation](https://github.com/sntcristian/MHEL-LLAMO/tree/b262dfb77285ea00993af8d453387fd7bf81228b). The upstream retrieval environment uses xlm-roberta-large and a FAISS index. Its four published WikiBELA files total 25.3 GB decimal before runtime caches. The reranker can use a saved candidate file without loading WikiBELA.

Prepare the upstream source and retriever data outside the checkout:

~~~sh
git clone https://github.com/sntcristian/MHEL-LLAMO "$work/mhel"
git -C "$work/mhel" checkout b262dfb77285ea00993af8d453387fd7bf81228b
hf download sntcristian/WikiBELA --revision 82b71b08c596b821e4a44c29a23d1c5fecd8165c --local-dir "$work/wikibela"
~~~

## Link mentions in your own documents

Pass a JSON array with `--documents`. Each document has a unique string or integer `id`, its original `text`, a `language` code, and an `entities` array. Each entity has an ID unique within the document, a `type`, and one or more mention ranges. The linker predicts Wikidata IDs separately for each occurrence. Ranges use zero-based Unicode character offsets with an exclusive end. An optional mention `text` must match the source slice. The authors’ prompts support `de`, `en`, `fi`, `fr`, `it`, `nl` and `sv`. Optional document strings `genre` and `publication_date` supply historical context.

~~~json
[{"id":7,"language":"en","text":"📰 Ada Lovelace wrote notes.","entities":[{"id":"ada","type":"PER","mentions":[{"start":2,"end":14}]}]}]
~~~

~~~sh
anno mhel --python "$bela_python" retrieve --upstream "$work/mhel" --model-dir "$work/wikibela" \
  --documents documents.json --device cpu --top-k 50 > candidates.json
anno mhel --python "$llm_python" rerank --upstream "$work/mhel" --model-dir "$local_chat_model" \
  --documents documents.json --candidates candidates.json --top-k 20 > links.jsonl
~~~

The candidate file contains one entry per supplied occurrence. You can also provide candidates from another retriever in this format. `wb_id` and `label` are required. Candidate `descr`, `type` and `min_date` provide context for disambiguation. A positive threshold requires a finite numeric `score` on each candidate, with the best candidate first.

~~~json
[{"doc_id":7,"entity_id":"ada","start":2,"end":14,"candidates":[{"wb_id":"Q7259","label":"Ada Lovelace","descr":"English mathematician"}]}]
~~~

Each output line includes the original `doc_id`, `entity_id`, `start`, `end`, source `text`, `type` and predicted `wikidata_id`. Different entities may share the same mention range. Unresolved occurrences have a null `wikidata_id`. For upstream-format datasets, both commands continue to accept `--dataset DIRECTORY` with UTF-8 `paragraphs_test.csv` and `annotations_test.csv` and their existing candidate format.

The authors' reported models include Mistral-Small-24B-Instruct-2501. The local chat checkpoint and candidate count are explicit choices. A positive score threshold accepts high-confidence retrievals directly. With threshold zero the model makes the NIL and candidate decisions.

You can also import released predictions without loading either model:

~~~sh
anno mhel import --dataset "$work/mhel/test_data/HIPE_FR" \
  --predictions "$work/mhel/results/HIPE_FR/mistral_24B_chain_k20_fr/output.csv"
~~~

The French import validated all 1,522 surface strings and offsets. The published predictions returned 801 correct IDs among 1,211 known mentions and correctly abstained on 232 of 311 NIL mentions. A CPU replay through `anno mhel rerank --documents` used SmolLM2-135M-Instruct, a caller-supplied candidate, and a document beginning with an emoji and a CRLF. The prompt run returned NIL for Ada Lovelace. The result retained integer document ID 7, entity ID `ada` and source offsets 3–15. Full WikiBELA retrieval and a fresh Mistral-24B run remain unverified here.

Across the 12 upstream test sets, the released tokenizer could represent 13,925 of 14,056 annotated occurrences. Of these, 1,035 required a shorter left context to keep the mention inside the 254-subword limit. Retrieval reports and skips annotations with missing source paragraphs (130 in the release). It raises an error identifying the occurrence if the mention alone exceeds the window. This happens for a single 788-character mention in German NewsEye. These counts measure input coverage before candidate scoring.
