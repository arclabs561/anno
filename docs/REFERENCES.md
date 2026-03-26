# References

Academic papers, datasets, and software cited across the anno codebase.

## Named Entity Recognition

- E. F. Tjong Kim Sang and F. De Meulder. "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition." *CoNLL*, 2003.
  [[PDF]](https://aclanthology.org/W03-0419.pdf)
  — Evaluation benchmark and span-level F1 definition used throughout anno-eval.

- R. Grishman and B. Sundheim. "Message Understanding Conference — 6: A Brief History." *COLING*, 1996.
  [[PDF]](https://aclanthology.org/C96-1079.pdf)
  — Original motivation for standardised NER evaluation.

- U. Zaratiana, N. Tomeh, P. Holat, and T. Charnois. "GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer." *NAACL*, 2024.
  [[arXiv:2311.08526]](https://arxiv.org/abs/2311.08526)
  — Architecture basis for the `gliner` and `gliner-candle` backends.

- U. Zaratiana, N. Tomeh, P. Holat, and T. Charnois. "GLiNER2: Multi-task Information Extraction with Generalist Models." 2025.
  [[arXiv:2507.18546]](https://arxiv.org/abs/2507.18546)
  — Basis for the `gliner2` multi-task backend (NER + classification + structure extraction).

- D. Bogdanov, A. Mokhov, et al. "NuNER: Entity Recognition Encoder Pre-training via LLM-Annotated Data." 2024.
  [[arXiv:2402.15343]](https://arxiv.org/abs/2402.15343)
  — Basis for the `nuner` zero-shot token-classification backend.

- J. Li, Y. Fei, et al. "Unified Named Entity Recognition as Word-Word Relation Classification." *AAAI*, 2022.
  [[arXiv:2112.10070]](https://arxiv.org/abs/2112.10070)
  — Basis for the `w2ner` backend (nested and discontinuous entities via handshaking matrix).

- J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019.
  [[arXiv:1810.04805]](https://arxiv.org/abs/1810.04805)
  — Underlying architecture for `bert-onnx`, `deberta-v3`, `albert`, and `candle-ner` backends.

- P. He, X. Liu, J. Gao, and W. Chen. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention." *ICLR*, 2021.
  [[arXiv:2006.03654]](https://arxiv.org/abs/2006.03654)
  — Architecture basis for the `deberta-v3` backend. Disentangled attention separately encodes content and position.

- Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." *ICLR*, 2020.
  [[arXiv:1909.11942]](https://arxiv.org/abs/1909.11942)
  — Architecture basis for the `albert` backend. Cross-layer parameter sharing and factorized embeddings.

## Classical Sequence Models

- J. Lafferty, A. McCallum, and F. Pereira. "Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data." *ICML*, 2001.
  [[PDF]](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers)
  — Basis for the `crf` backend.

- Z. Huang, W. Xu, and K. Yu. "Bidirectional LSTM-CRF Models for Sequence Tagging." 2015.
  [[arXiv:1508.01991]](https://arxiv.org/abs/1508.01991)
  — Foundational architecture for the `heuristic_crf` backend (CRF layer design).

- G. Lample, M. Ballesteros, S. Subramanian, K. Kawakami, and C. Dyer. "Neural Architectures for Named Entity Recognition." *NAACL-HLT*, 2016.
  [[PDF]](https://aclanthology.org/N16-1030.pdf)
  — BiLSTM-CRF with character-level embeddings; eliminates need for hand-crafted features.

- X. Ma and E. Hovy. "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF." *ACL*, 2016.
  [[arXiv:1603.01354]](https://arxiv.org/abs/1603.01354)
  — Combined character-level CNN + word-level BiLSTM + CRF pipeline for sequence labeling.

- L. R. Rabiner. "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition." *Proceedings of the IEEE* 77(2), 1989.
  [[PDF]](https://ieeexplore.ieee.org/document/18626)
  — Basis for the `hmm` backend.

- A. J. Viterbi. "Error bounds for convolutional codes and an asymptotically optimum decoding algorithm." *IEEE Transactions on Information Theory*, 1967.
  — Viterbi decoding algorithm used in HMM inference.

## Coreference Resolution

- K. Lee, L. He, M. Lewis, and L. Zettlemoyer. "End-to-end Neural Coreference Resolution." *EMNLP*, 2017.
  [[arXiv:1707.07045]](https://arxiv.org/abs/1707.07045)
  — Inspiration for the mention-ranking coreference architecture.

- K. Raghunathan, H. Lee, S. Rangarajan, N. Chambers, M. Surdeanu, D. Jurafsky, and C. Manning. "A Multi-Pass Sieve for Coreference Resolution." *EMNLP*, 2010.
  [[PDF]](https://aclanthology.org/D10-1048.pdf)
  — Basis for the rule-based sieve architecture in `SimpleCorefResolver`.

- S. Otmazgin, A. Cattan, and Y. Goldberg. "F-COREF: Fast, Accurate and Easy to Use Coreference Resolution." *AACL-IJCNLP*, 2022.
  [[arXiv:2209.04280]](https://arxiv.org/abs/2209.04280)
  — Basis for the `FCoref` neural coreference backend. LingMess mention detection with DistilRoBERTa.

- O. Bourgois and T. Poibeau. "Coreference Resolution for Machine Reading: A Survey." 2025.
  — Contemporary reference for the coreference approach in anno.

- D. Jurafsky and J. H. Martin. *Speech and Language Processing*, Ch. 21 (Coreference Resolution), 3rd ed. draft, 2024.
  [[Online]](https://web.stanford.edu/~jurafsky/slp3/)
  — Textbook reference for coreference fundamentals.

## Discourse Analysis

- B. J. Grosz, A. K. Joshi, and S. Weinstein. "Centering: A Framework for Modeling the Local Coherence of Discourse." *Computational Linguistics* 21(2), 1995.
  [[PDF]](https://aclanthology.org/J95-2003.pdf)
  — Theoretical basis for the `discourse::centering` module. Defines Cb, Cf, Cp, and transition types.

## Relation Extraction

- Y. Wang, Y. Yu, et al. "TPLinker: Single-stage Joint Extraction of Entities and Relations Through Token Pair Linking." *COLING*, 2020.
  [[arXiv:2010.13415]](https://arxiv.org/abs/2010.13415)
  — Architecture basis for the `tplinker` backend.

## Evaluation Frameworks and Datasets

- A. Akbik, T. Bergmann, D. Blythe, K. Rasul, S. Schweter, and R. Vollgraf. "FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP." *NAACL*, 2019.
  [[PDF]](https://aclanthology.org/N19-4010.pdf)
  — Referenced in the Scope section as an upstream training framework.

- O. Uzuner, B. R. South, S. Shen, and S. L. DuVall. "2010 i2b2/VA Challenge on Concepts, Assertions, and Relations in Clinical Text." *JAMIA*, 2011.
  — Motivates discontinuous entity support (clinical text has complex mention structures).

## Software

- HuggingFace Hub. <https://huggingface.co/> — model weight distribution and download.
- ONNX Runtime. <https://onnxruntime.ai/> — ML inference runtime used by the `onnx` feature.
- Candle (HuggingFace). <https://github.com/huggingface/candle> — pure-Rust ML framework used by the `candle` feature.
- lattix. <https://github.com/arclabs561/lattix> — graph/KG substrate used by `anno-graph`.
- muxer. <https://github.com/arclabs561/muxer> — randomised matrix sampler used by `anno-eval`.
- Oxigraph. <https://github.com/oxigraph/oxigraph> — recommended downstream RDF store for N-Triples export.
