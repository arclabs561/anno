# Historical Wikidata entity linking

MHEL-LLaMo takes identified mention occurrences and ranks Wikidata candidates. The release pairs WikiBELA retrieval with an instruction-tuned chat model. Use the retriever and reranker Python environments separately, following the [pinned authors’ installation](https://github.com/sntcristian/MHEL-LLAMO/tree/b262dfb77285ea00993af8d453387fd7bf81228b). The upstream retrieval environment uses xlm-roberta-large and a FAISS index. Its four published WikiBELA files total 25.3 GB decimal before runtime caches. The reranker can use a saved candidate file without loading WikiBELA.

Prepare the upstream source and retriever data outside the checkout:

~~~sh
git clone https://github.com/sntcristian/MHEL-LLAMO "$work/mhel"
git -C "$work/mhel" checkout b262dfb77285ea00993af8d453387fd7bf81228b
hf download sntcristian/WikiBELA --revision 82b71b08c596b821e4a44c29a23d1c5fecd8165c --local-dir "$work/wikibela"
~~~

Use an upstream-format dataset directory with UTF-8 paragraphs_test.csv and annotations_test.csv. The retriever takes original half-open Unicode character ranges, validates the surface string, and passes one supplied mention per context to the authors' model. Supported metadata languages are German, English, French, Italian, Dutch, Swedish and Finnish. The candidate JSON is suitable for the reranker. The predicted links are JSONL with one grounded occurrence per line. The model receives no reference Wikidata ID during reranking.

~~~sh
anno mhel --python "$bela_python" retrieve --upstream "$work/mhel" --model-dir "$work/wikibela" \
  --dataset "$dataset" --device cpu --top-k 50 > candidates.json
anno mhel --python "$llm_python" rerank --upstream "$work/mhel" --model-dir "$local_chat_model" \
  --dataset "$dataset" --candidates candidates.json --top-k 20 > links.jsonl
~~~

The authors' reported models include Mistral-Small-24B-Instruct-2501. The local chat checkpoint and candidate count are explicit choices. A positive score threshold accepts high-confidence retrievals directly. With threshold zero the model makes the NIL and candidate decisions. A NIL prediction becomes a JSON null Wikidata ID.

You can also import released predictions without loading either model:

~~~sh
anno mhel import --dataset "$work/mhel/test_data/HIPE_FR" \
  --predictions "$work/mhel/results/HIPE_FR/mistral_24B_chain_k20_fr/output.csv"
~~~

The French import validated all 1,522 surface strings and offsets. The published predictions returned 801 correct IDs among 1,211 known mentions and correctly abstained on 232 of 311 NIL mentions. A separate one-mention CPU run exercised the authors' two-step script with SmolLM2-135M-Instruct as an alternate local chat checkpoint. Full WikiBELA retrieval and a fresh Mistral-24B run remain unverified here.

Across the 12 upstream test sets, the released tokenizer could represent 13,925 of 14,056 annotated occurrences. Of these, 1,035 required a shorter left context to keep the mention inside the 254-subword limit. Retrieval reports and skips annotations with missing source paragraphs (130 in the release). It raises an error identifying the occurrence if the mention alone exceeds the window. This happens for a single 788-character mention in German NewsEye. These counts measure input coverage before candidate scoring.
