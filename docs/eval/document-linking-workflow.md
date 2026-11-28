# From raw text to document relations and Wikidata links

For a local workflow, run named-entity recognition and CorPipe on exactly the same UTF-8 text. The converter joins their character ranges and emits the shared JSON document format accepted by GLiDRE and MHEL. The source object carries the caller’s `id`, original `text` and `language`.

~~~sh
jq -rj '.text' source.json > raw.txt
anno extract --model bert-onnx --file raw.txt --format json > ner.json
anno corpipe --python "$corpipe_python" --predictor "$corpipe_predictor" \
  --checkpoint "$corpipe_model" --tokenizer-model "$udpipe_model" < raw.txt > coref.json
python scripts/prepare_linking_document.py source.json ner.json coref.json > mentions.json
anno glidre --python "$glidre_python" --upstream "$glidre_source" --model-dir "$glidre_model" \
  --input mentions.json --schema predicates.json > relations.json
anno mhel --python "$mhel_python" rerank --upstream "$mhel_source" --model-dir "$chat_model" \
  --documents mentions.json --candidates candidates.json > links.jsonl
~~~

The converter validates the source digests of both outputs and checks each overt mention against its half-open Unicode character range. A CorPipe cluster enters the result when any of its overt spans exactly matches a named-entity span. It keeps the cluster’s other overt mentions, takes its type from the highest-confidence exact match, and records the original cluster ID and matched NER IDs. Unclustered NER detections become singleton entities. Equal names at different offsets remain distinct unless CorPipe joins them. Implicit CorefUD nodes have no source span. Unanchored clusters are outside this named-entity workflow.

GLiDRE scores directed relations between the supplied entities. MHEL scores each mention independently against caller-supplied Wikidata candidates. Its occurrence keys are document ID, entity ID, start and end. To acquire candidates with WikiBELA, replace the candidate source with `anno mhel retrieve --documents mentions.json` and provide the retriever artifacts. [GLiDRE setup](../models/glidre.md) · [MHEL candidate format and setup](../models/mhel.md) · [CorPipe setup](../models/corpipe.md)

## Authored CPU replay

The two English passages contain a Unicode prefix, repeated names, pronouns and two explicitly stated facts. The review covers named mentions and personal pronouns. Each occurrence received the same seven Wikidata candidates in the same order, including two distractors. Each document received the same three predicate prompts. Reviewed Wikidata IDs and relations entered only during scoring. The comparison supplied reviewed mention clusters to the second run and kept both downstream models and their settings fixed. These cases measure stage behavior on authored text with a closed candidate list. [Input documents](../../scripts/qa/document_linking/documents.json) · [Independent review](../../scripts/qa/document_linking/reviewed.json) · [Shared candidates](../../scripts/qa/document_linking/candidates.json) · [Predicate schema](../../scripts/qa/document_linking/schema.json)

BERT recovered all ten reviewed names with the correct types. CorPipe linked all five reviewed entities’ repeated mentions and pronouns; its anchored clusters also included “a citizen of Barbados” and “the capital of France”. Both descriptions refer to their cluster’s named entity. They are additional overt mentions beyond the names and pronouns chosen for review.

The model-produced mentions raised Rihanna’s citizenship score from 0.175 to 0.305, crossing the fixed GLiDRE threshold of 0.3. France’s capital relation scored 0.125 with model-produced clusters and 0.100 with reviewed clusters. All four required endpoint entities were present in both arms. The capital omission therefore occurred during relation scoring. The two additional descriptive mentions explain how the inputs differed.

MHEL’s 135M chat model returned zero correct Wikidata IDs on the twelve reviewed occurrences in the model-produced arm (seven NIL, five incorrect IDs). With reviewed clusters it returned one correct, nine NIL and two incorrect. All five correct Wikidata entities were in the candidate list in both runs. Six of twelve shared occurrences received different outcomes despite matching surface, context and candidate prompts. The reranker samples completions, and these runs do not isolate a clustering effect on linking accuracy. [Per-stage counts and scores](../../scripts/qa/document_linking/receipt.json)

### Replay

The study runner accepts local paths for all four providers, keeps raw model outputs in a work directory, and writes separate artifacts for each stage. The scorer reads the saved outputs and the separate review. GLiDRE’s operating threshold is 0.3. A second run at threshold zero records the reference facts’ underlying scores. MHEL uses threshold zero and the checkpoint’s original generation settings. The installed Transformers pipeline samples token completions, so individual reranker decisions can vary between invocations.

~~~sh
export BERT_ONNX_MODEL_PATH="$bert_dir" ANNO_NO_DOWNLOADS=1 HF_HUB_OFFLINE=1
python scripts/qa/document_linking/run.py --output "$work" --anno "$anno" \
  --corpipe-python "$corpipe_python" --corpipe-predictor "$corpipe_predictor" \
  --corpipe-model "$corpipe_model" --udpipe "$udpipe_model" --corpipe-cache "$corpipe_hub" \
  --glidre-python "$glidre_python" --glidre-upstream "$glidre_source" --glidre-model "$glidre_model" \
  --mhel-python "$mhel_python" --mhel-upstream "$mhel_source" --mhel-model "$chat_model"
python scripts/qa/document_linking/score.py "$work"
~~~

The CPU run used BERT-base-NER ONNX, CorPipe 26 base with English EWT tokenization, GLiDRE multilingual v2, and the MHEL prompt chain with SmolLM2-135M-Instruct. A fresh WikiBELA search or a 24B reranker would be separate experiments. [BERT checkpoint](https://huggingface.co/protectai/bert-base-NER-onnx/tree/84d28ab2bc28401b4353b26033eb043c8c8a0739) · [CorPipe checkpoint](https://huggingface.co/ufal/corpipe26-onestage-corefud1.4-base-260702/tree/cb62351dd97b58c95bb7e12258a669e471cb577d) · [GLiDRE checkpoint](https://huggingface.co/cea-list-ia/glidre_multi_v2/tree/1881f8126342d47ac4b9b1c7049d4af99424d8f3) · [MHEL code](https://github.com/sntcristian/MHEL-LLAMO/tree/b262dfb77285ea00993af8d453387fd7bf81228b) · [Chat model checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac)
