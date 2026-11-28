// Adapted from gliner 0.0.19. See gliner-LICENSE. Changes: pinned browser imports and tokenizer revision, guarded sequence length.
import { env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
import ort_CPU from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.mjs';
import ort_WEBGPU from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.mjs';
import ort_WEBGL from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.wasm.min.mjs';

var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);

// src/lib/model.ts
var Model = class {
  constructor(config, processor, decoder, onnxWrapper) {
    this.config = config;
    this.processor = processor;
    this.decoder = decoder;
    this.onnxWrapper = onnxWrapper;
  }
  async initialize() {
    await this.onnxWrapper.init();
  }
};
var SpanModel = class extends Model {
  prepareInputs(batch) {
    const batch_size = batch.inputsIds.length;
    const num_tokens = batch.inputsIds[0].length;
    const num_spans = batch.spanIdxs[0].length;
    const createTensor = (data, shape, tensorType = "int64") => {
      return new this.onnxWrapper.ort.Tensor(tensorType, data.flat(Infinity), shape);
    };
    let input_ids = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask = createTensor(batch.attentionMasks, [batch_size, num_tokens]);
    let words_mask = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths = createTensor(batch.textLengths, [batch_size, 1]);
    let span_idx = createTensor(batch.spanIdxs, [batch_size, num_spans, 2]);
    let span_mask = createTensor(batch.spanMasks, [batch_size, num_spans], "bool");
    const feeds = {
      input_ids,
      attention_mask,
      words_mask,
      text_lengths,
      span_idx,
      span_mask
    };
    return feeds;
  }
  async inference(texts, entities, flatNer = false, threshold = 0.5, multiLabel = false) {
    let batch = this.processor.prepareBatch(texts, entities);
    if(batch.inputsIds[0].length>512)throw new Error("Passage and labels exceed the 512-token context. Shorten the passage.");
    let feeds = this.prepareInputs(batch);
    const results = await this.onnxWrapper.run(feeds);
    const modelOutput = results["logits"].data;
    const batchSize = batch.batchTokens.length;
    const inputLength = Math.max(...batch.textLengths);
    const maxWidth = this.config.max_width;
    const numEntities = entities.length;
    const batchIds = Array.from({ length: batchSize }, (_, i) => i);
    const decodedSpans = this.decoder.decode(
      batchSize,
      inputLength,
      maxWidth,
      numEntities,
      texts,
      batchIds,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel
    );
    return decodedSpans;
  }
  async inference_with_chunking(texts, entities, flatNer = false, threshold = 0.5, multiLabel = false, batch_size = 4, max_words = 512) {
    const {
      idToClass
    } = this.processor.createMappings(entities);
    let batchIds = [];
    let batchTokens = [];
    let batchWordsStartIdx = [];
    let batchWordsEndIdx = [];
    texts.forEach((text, id) => {
      let [tokens, wordsStartIdx, wordsEndIdx] = this.processor.tokenizeText(text);
      let num_sub_batches = Math.ceil(tokens.length / max_words);
      for (let i = 0; i < num_sub_batches; i++) {
        let start = i * max_words;
        let end = Math.min((i + 1) * max_words, tokens.length);
        batchIds.push(id);
        batchTokens.push(tokens.slice(start, end));
        batchWordsStartIdx.push(wordsStartIdx.slice(start, end));
        batchWordsEndIdx.push(wordsEndIdx.slice(start, end));
      }
    });
    let num_batches = Math.ceil(batchIds.length / batch_size);
    let finalDecodedSpans = [];
    for (let id = 0; id < texts.length; id++) {
      finalDecodedSpans.push([]);
    }
    for (let batch_id = 0; batch_id < num_batches; batch_id++) {
      let start = batch_id * batch_size;
      let end = Math.min((batch_id + 1) * batch_size, batchIds.length);
      let currBatchTokens = batchTokens.slice(start, end);
      let currBatchWordsStartIdx = batchWordsStartIdx.slice(start, end);
      let currBatchWordsEndIdx = batchWordsEndIdx.slice(start, end);
      let currBatchIds = batchIds.slice(start, end);
      let [inputTokens, textLengths, promptLengths] = this.processor.prepareTextInputs(currBatchTokens, entities);
      let [inputsIds, attentionMasks, wordsMasks] = this.processor.encodeInputs(inputTokens, promptLengths);
      inputsIds = this.processor.padArray(inputsIds);
      attentionMasks = this.processor.padArray(attentionMasks);
      wordsMasks = this.processor.padArray(wordsMasks);
      let { spanIdxs, spanMasks } = this.processor.prepareSpans(currBatchTokens, this.config["max_width"]);
      spanIdxs = this.processor.padArray(spanIdxs, 3);
      spanMasks = this.processor.padArray(spanMasks);
      let batch = {
        inputsIds,
        attentionMasks,
        wordsMasks,
        textLengths,
        spanIdxs,
        spanMasks,
        idToClass,
        batchTokens: currBatchTokens,
        batchWordsStartIdx: currBatchWordsStartIdx,
        batchWordsEndIdx: currBatchWordsEndIdx
      };
      let feeds = this.prepareInputs(batch);
      const results = await this.onnxWrapper.run(feeds);
      const modelOutput = results["logits"].data;
      const batchSize = batch.batchTokens.length;
      const inputLength = Math.max(...batch.textLengths);
      const maxWidth = this.config.max_width;
      const numEntities = entities.length;
      const decodedSpans = this.decoder.decode(
        batchSize,
        inputLength,
        maxWidth,
        numEntities,
        texts,
        currBatchIds,
        batch.batchWordsStartIdx,
        batch.batchWordsEndIdx,
        batch.idToClass,
        modelOutput,
        flatNer,
        threshold,
        multiLabel
      );
      for (let i = 0; i < currBatchIds.length; i++) {
        const originalTextId = currBatchIds[i];
        finalDecodedSpans[originalTextId].push(...decodedSpans[i]);
      }
    }
    return finalDecodedSpans;
  }
};
var TokenModel = class extends Model {
  prepareInputs(batch) {
    const batch_size = batch.inputsIds.length;
    const num_tokens = batch.inputsIds[0].length;
    const createTensor = (data, shape, tensorType = "int64") => {
      return new this.onnxWrapper.ort.Tensor(tensorType, data.flat(Infinity), shape);
    };
    let input_ids = createTensor(batch.inputsIds, [batch_size, num_tokens]);
    let attention_mask = createTensor(batch.attentionMasks, [batch_size, num_tokens]);
    let words_mask = createTensor(batch.wordsMasks, [batch_size, num_tokens]);
    let text_lengths = createTensor(batch.textLengths, [batch_size, 1]);
    const feeds = {
      input_ids,
      attention_mask,
      words_mask,
      text_lengths
    };
    return feeds;
  }
  async inference(texts, entities, flatNer = false, threshold = 0.5, multiLabel = false) {
    let batch = this.processor.prepareBatch(texts, entities);
    let feeds = this.prepareInputs(batch);
    const results = await this.onnxWrapper.run(feeds);
    const modelOutput = results["logits"].data;
    const batchSize = batch.batchTokens.length;
    const inputLength = Math.max(...batch.textLengths);
    const numEntities = entities.length;
    const batchIds = Array.from({ length: batchSize }, (_, i) => i);
    const decodedSpans = this.decoder.decode(
      batchSize,
      inputLength,
      numEntities,
      texts,
      batchIds,
      batch.batchWordsStartIdx,
      batch.batchWordsEndIdx,
      batch.idToClass,
      modelOutput,
      flatNer,
      threshold,
      multiLabel
    );
    return decodedSpans;
  }
  async inference_with_chunking(texts, entities, flatNer = true, threshold = 0.5, multiLabel = false, batch_size = 4, max_words = 512) {
    const {
      idToClass
    } = this.processor.createMappings(entities);
    let batchIds = [];
    let batchTokens = [];
    let batchWordsStartIdx = [];
    let batchWordsEndIdx = [];
    texts.forEach((text, id) => {
      let [tokens, wordsStartIdx, wordsEndIdx] = this.processor.tokenizeText(text);
      let num_sub_batches = Math.ceil(tokens.length / max_words);
      for (let i = 0; i < num_sub_batches; i++) {
        let start = i * max_words;
        let end = Math.min((i + 1) * max_words, tokens.length);
        batchIds.push(id);
        batchTokens.push(tokens.slice(start, end));
        batchWordsStartIdx.push(wordsStartIdx.slice(start, end));
        batchWordsEndIdx.push(wordsEndIdx.slice(start, end));
      }
    });
    let num_batches = Math.ceil(batchIds.length / batch_size);
    let finalDecodedSpans = [];
    for (let id = 0; id < texts.length; id++) {
      finalDecodedSpans.push([]);
    }
    for (let batch_id = 0; batch_id < num_batches; batch_id++) {
      let start = batch_id * batch_size;
      let end = Math.min((batch_id + 1) * batch_size, batchIds.length);
      let currBatchTokens = batchTokens.slice(start, end);
      let currBatchWordsStartIdx = batchWordsStartIdx.slice(start, end);
      let currBatchWordsEndIdx = batchWordsEndIdx.slice(start, end);
      let currBatchIds = batchIds.slice(start, end);
      let [inputTokens, textLengths, promptLengths] = this.processor.prepareTextInputs(currBatchTokens, entities);
      let [inputsIds, attentionMasks, wordsMasks] = this.processor.encodeInputs(inputTokens, promptLengths);
      inputsIds = this.processor.padArray(inputsIds);
      attentionMasks = this.processor.padArray(attentionMasks);
      wordsMasks = this.processor.padArray(wordsMasks);
      let batch = {
        inputsIds,
        attentionMasks,
        wordsMasks,
        textLengths,
        idToClass,
        batchTokens: currBatchTokens,
        batchWordsStartIdx: currBatchWordsStartIdx,
        batchWordsEndIdx: currBatchWordsEndIdx
      };
      let feeds = this.prepareInputs(batch);
      const results = await this.onnxWrapper.run(feeds);
      const modelOutput = results["logits"].data;
      const batchSize = batch.batchTokens.length;
      const inputLength = Math.max(...batch.textLengths);
      const numEntities = entities.length;
      const decodedSpans = this.decoder.decode(
        batchSize,
        inputLength,
        numEntities,
        texts,
        currBatchIds,
        batch.batchWordsStartIdx,
        batch.batchWordsEndIdx,
        batch.idToClass,
        modelOutput,
        flatNer,
        threshold,
        multiLabel
      );
      for (let i = 0; i < currBatchIds.length; i++) {
        const originalTextId = currBatchIds[i];
        finalDecodedSpans[originalTextId].push(...decodedSpans[i]);
      }
    }
    return finalDecodedSpans;
  }
};

// src/lib/processor.ts
var WhitespaceTokenSplitter = class {
  constructor() {
    __publicField(this, "whitespacePattern");
    this.whitespacePattern = /\w+(?:[-_]\w+)*|\S/g;
  }
  *call(text) {
    let match;
    while ((match = this.whitespacePattern.exec(text)) !== null) {
      yield [match[0], match.index, this.whitespacePattern.lastIndex];
    }
  }
};
var Processor = class {
  constructor(config, tokenizer, wordsSplitter) {
    __publicField(this, "config");
    __publicField(this, "tokenizer");
    __publicField(this, "wordsSplitter");
    this.config = config;
    this.tokenizer = tokenizer;
    this.wordsSplitter = wordsSplitter;
  }
  tokenizeText(text) {
    let tokens = [];
    let wordsStartIdx = [];
    let wordsEndIdx = [];
    for (const [token, start, end] of this.wordsSplitter.call(text)) {
      tokens.push(token);
      wordsStartIdx.push(start);
      wordsEndIdx.push(end);
    }
    return [tokens, wordsStartIdx, wordsEndIdx];
  }
  batchTokenizeText(texts) {
    let batchTokens = [];
    let batchWordsStartIdx = [];
    let batchWordsEndIdx = [];
    for (const text of texts) {
      const [tokens, wordsStartIdx, wordsEndIdx] = this.tokenizeText(text);
      batchTokens.push(tokens);
      batchWordsStartIdx.push(wordsStartIdx);
      batchWordsEndIdx.push(wordsEndIdx);
    }
    return [batchTokens, batchWordsStartIdx, batchWordsEndIdx];
  }
  createMappings(classes) {
    const classToId = {};
    const idToClass = {};
    classes.forEach((className, index) => {
      const id = index + 1;
      classToId[className] = id;
      idToClass[id] = className;
    });
    return { classToId, idToClass };
  }
  prepareTextInputs(tokens, entities) {
    const inputTexts = [];
    const promptLengths = [];
    const textLengths = [];
    tokens.forEach((text) => {
      textLengths.push(text.length);
      let inputText = [];
      for (let ent of entities) {
        inputText.push("<<ENT>>");
        inputText.push(ent);
      }
      inputText.push("<<SEP>>");
      const promptLength = inputText.length;
      promptLengths.push(promptLength);
      inputText = inputText.concat(text);
      inputTexts.push(inputText);
    });
    return [inputTexts, textLengths, promptLengths];
  }
  encodeInputs(texts, promptLengths = null) {
    let wordsMasks = [];
    let inputsIds = [];
    let attentionMasks = [];
    for (let id = 0; id < texts.length; id++) {
      let promptLength = promptLengths ? promptLengths[id] : 0;
      let tokenizedInputs = texts[id];
      let wordsMask = [0];
      let inputIds = [1];
      let attentionMask = [1];
      let c = 1;
      tokenizedInputs.forEach((word, wordId) => {
        let wordTokens = this.tokenizer.encode(word).slice(1, -1);
        wordTokens.forEach((token, tokenId) => {
          attentionMask.push(1);
          if (wordId < promptLength) {
            wordsMask.push(0);
          } else if (tokenId === 0) {
            wordsMask.push(c);
            c++;
          } else {
            wordsMask.push(0);
          }
          inputIds.push(token);
        });
      });
      wordsMask.push(0);
      inputIds.push(this.tokenizer.sep_token_id);
      attentionMask.push(1);
      wordsMasks.push(wordsMask);
      inputsIds.push(inputIds);
      attentionMasks.push(attentionMask);
    }
    return [inputsIds, attentionMasks, wordsMasks];
  }
  padArray(arr, dimensions = 2) {
    if (dimensions < 2 || dimensions > 3) {
      throw new Error("Only 2D and 3D arrays are supported");
    }
    const maxLength = Math.max(...arr.map((subArr) => subArr.length));
    const finalDim = dimensions === 3 ? arr[0][0].length : 0;
    return arr.map((subArr) => {
      const padCount = maxLength - subArr.length;
      const padding = Array(padCount).fill(dimensions === 3 ? Array(finalDim).fill(0) : 0);
      return [...subArr, ...padding];
    });
  }
};
var SpanProcessor = class extends Processor {
  constructor(config, tokenizer, wordsSplitter) {
    super(config, tokenizer, wordsSplitter);
  }
  prepareSpans(batchTokens, maxWidth = 12) {
    let spanIdxs = [];
    let spanMasks = [];
    batchTokens.forEach((tokens) => {
      let textLength = tokens.length;
      let spanIdx = [];
      let spanMask = [];
      for (let i = 0; i < textLength; i++) {
        for (let j = 0; j < maxWidth; j++) {
          let endIdx = Math.min(i + j, textLength - 1);
          spanIdx.push([i, endIdx]);
          spanMask.push(endIdx < textLength ? true : false);
        }
      }
      spanIdxs.push(spanIdx);
      spanMasks.push(spanMask);
    });
    return { spanIdxs, spanMasks };
  }
  prepareBatch(texts, entities) {
    const [batchTokens, batchWordsStartIdx, batchWordsEndIdx] = this.batchTokenizeText(texts);
    const { idToClass } = this.createMappings(entities);
    const [inputTokens, textLengths, promptLengths] = this.prepareTextInputs(batchTokens, entities);
    let [inputsIds, attentionMasks, wordsMasks] = this.encodeInputs(inputTokens, promptLengths);
    inputsIds = this.padArray(inputsIds);
    attentionMasks = this.padArray(attentionMasks);
    wordsMasks = this.padArray(wordsMasks);
    let { spanIdxs, spanMasks } = this.prepareSpans(batchTokens, this.config["max_width"]);
    spanIdxs = this.padArray(spanIdxs, 3);
    spanMasks = this.padArray(spanMasks);
    return {
      inputsIds,
      attentionMasks,
      wordsMasks,
      textLengths,
      spanIdxs,
      spanMasks,
      idToClass,
      batchTokens,
      batchWordsStartIdx,
      batchWordsEndIdx
    };
  }
};
var TokenProcessor = class extends Processor {
  constructor(config, tokenizer, wordsSplitter) {
    super(config, tokenizer, wordsSplitter);
  }
  prepareBatch(texts, entities) {
    const [batchTokens, batchWordsStartIdx, batchWordsEndIdx] = this.batchTokenizeText(texts);
    const { idToClass } = this.createMappings(entities);
    const [inputTokens, textLengths, promptLengths] = this.prepareTextInputs(batchTokens, entities);
    let [inputsIds, attentionMasks, wordsMasks] = this.encodeInputs(inputTokens, promptLengths);
    inputsIds = this.padArray(inputsIds);
    attentionMasks = this.padArray(attentionMasks);
    wordsMasks = this.padArray(wordsMasks);
    return {
      inputsIds,
      attentionMasks,
      wordsMasks,
      textLengths,
      idToClass,
      batchTokens,
      batchWordsStartIdx,
      batchWordsEndIdx
    };
  }
};

// src/lib/decoder.ts
var isNested = (idx1, idx2) => {
  return idx1[0] <= idx2[0] && idx1[1] >= idx2[1] || idx2[0] <= idx1[0] && idx2[1] >= idx1[1];
};
var hasOverlapping = (idx1, idx2, multiLabel = false) => {
  if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
    return !multiLabel;
  }
  if (idx1[0] > idx2[1] || idx2[0] > idx1[1]) {
    return false;
  }
  return true;
};
var hasOverlappingNested = (idx1, idx2, multiLabel = false) => {
  if (idx1.slice(0, 2).toString() === idx2.slice(0, 2).toString()) {
    return !multiLabel;
  }
  if (idx1[0] > idx2[1] || idx2[0] > idx1[1] || isNested(idx1, idx2)) {
    return false;
  }
  return true;
};
var sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};
var BaseDecoder = class _BaseDecoder {
  constructor(config) {
    __publicField(this, "config");
    if (new.target === _BaseDecoder) {
      throw new TypeError("Cannot instantiate an abstract class.");
    }
    this.config = config;
  }
  greedySearch(spans, flatNer = true, multiLabel = false) {
    const hasOv = flatNer ? (idx1, idx2) => hasOverlapping(idx1, idx2, multiLabel) : (idx1, idx2) => hasOverlappingNested(idx1, idx2, multiLabel);
    const newList = [];
    const spanProb = spans.slice().sort((a, b) => b[b.length - 1] - a[a.length - 1]);
    for (let i = 0; i < spanProb.length; i++) {
      const b = spanProb[i];
      let flag = false;
      for (const newSpan of newList) {
        if (hasOv(b.slice(1, 3), newSpan.slice(1, 3))) {
          flag = true;
          break;
        }
      }
      if (!flag) {
        newList.push(b);
      }
    }
    return newList.sort((a, b) => a[1] - b[1]);
  }
};
var SpanDecoder = class extends BaseDecoder {
  decode(batchSize, inputLength, maxWidth, numEntities, texts, batchIds, batchWordsStartIdx, batchWordsEndIdx, idToClass, modelOutput, flatNer = false, threshold = 0.5, multiLabel = false) {
    const spans = [];
    for (let batch = 0; batch < batchSize; batch++) {
      spans.push([]);
    }
    const batchPadding = inputLength * maxWidth * numEntities;
    const startTokenPadding = maxWidth * numEntities;
    const endTokenPadding = numEntities * 1;
    modelOutput.forEach((value, id) => {
      let batch = Math.floor(id / batchPadding);
      let startToken = Math.floor(id / startTokenPadding) % inputLength;
      let endToken = startToken + Math.floor(id / endTokenPadding) % maxWidth;
      let entity = id % numEntities;
      let prob = sigmoid(value);
      if (prob >= threshold && startToken < batchWordsStartIdx[batch].length && endToken < batchWordsEndIdx[batch].length) {
        let globalBatch = batchIds[batch];
        let startIdx = batchWordsStartIdx[batch][startToken];
        let endIdx = batchWordsEndIdx[batch][endToken];
        let spanText = texts[globalBatch].slice(startIdx, endIdx);
        spans[batch].push([spanText, startIdx, endIdx, idToClass[entity + 1], prob]);
      }
    });
    const allSelectedSpans = [];
    spans.forEach((resI, id) => {
      const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
      allSelectedSpans.push(selectedSpans);
    });
    return allSelectedSpans;
  }
};
var TokenDecoder = class extends BaseDecoder {
  decode(batchSize, inputLength, numEntities, texts, batchIds, batchWordsStartIdx, batchWordsEndIdx, idToClass, modelOutput, flatNer = false, threshold = 0.5, multiLabel = false) {
    const positionPadding = batchSize * inputLength * numEntities;
    const batchPadding = inputLength * numEntities;
    const tokenPadding = numEntities;
    let selectedStarts = [];
    let selectedEnds = [];
    let insideScore = [];
    for (let id = 0; id < batchSize; id++) {
      selectedStarts.push([]);
      selectedEnds.push([]);
      let batches = [];
      for (let j = 0; j < inputLength; j++) {
        let sequence = Array(numEntities).fill(0);
        batches.push(sequence);
      }
      insideScore.push(batches);
    }
    modelOutput.forEach((value, id) => {
      let position = Math.floor(id / positionPadding);
      let batch = Math.floor(id / batchPadding) % batchSize;
      let token = Math.floor(id / tokenPadding) % inputLength;
      let entity = id % numEntities;
      let prob = sigmoid(value);
      if (prob >= threshold && token < batchWordsEndIdx[batch].length) {
        if (position == 0) {
          selectedStarts[batch].push([token, entity]);
        } else if (position == 1) {
          selectedEnds[batch].push([token, entity]);
        }
      }
      if (position == 2) {
        insideScore[batch][token][entity] = prob;
      }
    });
    const spans = [];
    for (let batch = 0; batch < batchSize; batch++) {
      let batchSpans = [];
      for (let [start, clsSt] of selectedStarts[batch]) {
        for (let [end, clsEd] of selectedEnds[batch]) {
          if (end >= start && clsSt === clsEd) {
            const insideSpanScores = insideScore[batch].slice(start, end + 1).map((tokenScores) => tokenScores[clsSt]);
            if (insideSpanScores.some((score) => score < threshold)) continue;
            const spanScore = insideSpanScores.reduce((a, b) => a + b, 0) / insideSpanScores.length;
            let startIdx = batchWordsStartIdx[batch][start];
            let endIdx = batchWordsEndIdx[batch][end];
            let spanText = texts[batchIds[batch]].slice(startIdx, endIdx);
            batchSpans.push([spanText, startIdx, endIdx, idToClass[clsSt + 1], spanScore]);
          }
        }
      }
      spans.push(batchSpans);
    }
    const allSelectedSpans = [];
    spans.forEach((resI, id) => {
      const selectedSpans = this.greedySearch(resI, flatNer, multiLabel);
      allSelectedSpans.push(selectedSpans);
    });
    return allSelectedSpans;
  }
};
var ONNX_WASM_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
var DEFAULT_WASM_PATHS = ONNX_WASM_CDN_URL;
var ONNXWebWrapper = class {
  constructor(settings) {
    this.settings = settings;
    __publicField(this, "ort", ort_CPU);
    __publicField(this, "session", null);
    const { executionProvider, wasmPaths = DEFAULT_WASM_PATHS } = settings;
    if (!executionProvider) {
      this.settings.executionProvider = "webgpu";
    }
    switch (this.settings.executionProvider) {
      case "cpu":
      case "wasm":
        break;
      case "webgpu":
        this.ort = ort_WEBGPU;
        break;
      case "webgl":
        this.ort = ort_WEBGL;
        break;
      default:
        throw new Error(`ONNXWrapper: Invalid execution provider: '${executionProvider}'`);
    }
    this.ort.env.wasm.wasmPaths = wasmPaths;
  }
  async init() {
    if (!this.session) {
      const { modelPath, executionProvider, fetchBinary, multiThread } = this.settings;
      if (executionProvider === "cpu" || executionProvider === "wasm") {
        if (fetchBinary) {
          const binaryURL = this.ort.env.wasm.wasmPaths + "ort-wasm-simd-threaded.wasm";
          const response = await fetch(binaryURL);
          const binary = await response.arrayBuffer();
          this.ort.env.wasm.wasmBinary = binary;
        }
        if (multiThread) {
          const maxPossibleThreads = navigator.hardwareConcurrency ?? 0;
          const maxThreads = Math.min(
            this.settings.maxThreads ?? maxPossibleThreads,
            maxPossibleThreads
          );
          this.ort.env.wasm.numThreads = maxThreads;
        }
      }
      this.session = await this.ort.InferenceSession.create(modelPath, {
        executionProviders: [executionProvider]
      });
    }
  }
  async run(feeds, options = {}) {
    if (!this.session) {
      throw new Error("ONNXWrapper: Session not initialized. Please call init() first.");
    }
    return await this.session.run(feeds, options);
  }
};

// src/web/Gliner.ts
var Gliner = class {
  constructor(config) {
    this.config = config;
    __publicField(this, "model", null);
    env.allowLocalModels = config.transformersSettings?.allowLocalModels ?? false;
    env.useBrowserCache = config.transformersSettings?.useBrowserCache ?? false;
    this.config = {
      ...config,
      maxWidth: config.maxWidth || 12,
      modelType: config.modelType || "span-level"
    };
  }
  async initialize() {
    const { tokenizerPath, onnxSettings, maxWidth } = this.config;
    const tokenizer = await AutoTokenizer.from_pretrained(tokenizerPath,{revision:this.config.tokenizerRevision});
    const wordSplitter = new WhitespaceTokenSplitter();
    const onnxWrapper = new ONNXWebWrapper(onnxSettings);
    if (this.config.modelType == "span-level") {
      const processor = new SpanProcessor({ max_width: maxWidth }, tokenizer, wordSplitter);
      const decoder = new SpanDecoder({ max_width: maxWidth });
      this.model = new SpanModel({ max_width: maxWidth }, processor, decoder, onnxWrapper);
    } else {
      const processor = new TokenProcessor({ max_width: maxWidth }, tokenizer, wordSplitter);
      const decoder = new TokenDecoder({ max_width: maxWidth });
      this.model = new TokenModel({ max_width: maxWidth }, processor, decoder, onnxWrapper);
    }
    await this.model.initialize();
  }
  async inference({
    texts,
    entities,
    flatNer = true,
    threshold = 0.5,
    multiLabel = false
  }) {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }
    const result = await this.model.inference(texts, entities, flatNer, threshold, multiLabel);
    return this.mapRawResultToResponse(result);
  }
  async inference_with_chunking({
    texts,
    entities,
    flatNer = false,
    threshold = 0.5
  }) {
    if (!this.model) {
      throw new Error("Model is not initialized. Call initialize() first.");
    }
    const result = await this.model.inference_with_chunking(texts, entities, flatNer, threshold);
    return this.mapRawResultToResponse(result);
  }
  mapRawResultToResponse(rawResult) {
    const response = [];
    for (const individualResult of rawResult) {
      const entityResult = individualResult.map(
        ([spanText, start, end, label, score]) => ({
          spanText,
          start,
          end,
          label,
          score
        })
      );
      response.push(entityResult);
    }
    return response;
  }
};

export { Gliner, WhitespaceTokenSplitter };
