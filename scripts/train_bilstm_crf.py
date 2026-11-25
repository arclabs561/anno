#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.6.0",
#     "datasets>=2.14.0",
#     "transformers>=4.30.0",
#     "tqdm>=4.65.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Train BiLSTM-CRF NER model on CoNLL-2003.

This script trains a BiLSTM-CRF model for NER using PyTorch and exports
the model to ONNX format for use in the Rust backend.

The architecture follows Lample et al. (2016) "Neural Architectures for NER":
- Word embeddings (GloVe or random)
- Character-level CNN embeddings
- Bidirectional LSTM
- CRF decoder

Usage:
    uv run scripts/train_bilstm_crf.py

Output:
    - bilstm_crf_ner.onnx: ONNX model for Rust inference
    - bilstm_crf_vocab.json: Vocabulary and tag mappings
    - bilstm_crf_weights.json: CRF transition weights for fallback
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# NER tag set (BIO scheme)
TAG_TO_IDX = {
    'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8,
    '<PAD>': 9, '<START>': 10, '<END>': 11
}
IDX_TO_TAG = {v: k for k, v in TAG_TO_IDX.items()}
NUM_TAGS = len(TAG_TO_IDX)

# Special tokens
PAD_IDX = 0
UNK_IDX = 1


class NERDataset(Dataset):
    """PyTorch dataset for NER."""
    
    def __init__(self, sentences, word_to_idx, max_len=128):
        self.sentences = sentences
        self.word_to_idx = word_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words, tags = self.sentences[idx]
        
        # Truncate
        words = words[:self.max_len]
        tags = tags[:self.max_len]
        
        # Convert to indices
        word_ids = [self.word_to_idx.get(w.lower(), UNK_IDX) for w in words]
        tag_ids = [TAG_TO_IDX[t] for t in tags]
        
        return {
            'words': torch.tensor(word_ids, dtype=torch.long),
            'tags': torch.tensor(tag_ids, dtype=torch.long),
            'length': len(words)
        }


def collate_fn(batch):
    """Pad sequences to same length."""
    max_len = max(item['length'] for item in batch)
    
    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_ids = torch.full((len(batch), max_len), TAG_TO_IDX['<PAD>'], dtype=torch.long)
    lengths = []
    
    for i, item in enumerate(batch):
        length = item['length']
        word_ids[i, :length] = item['words']
        tag_ids[i, :length] = item['tags']
        lengths.append(length)
    
    return {
        'words': word_ids,
        'tags': tag_ids,
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }


class CRF(nn.Module):
    """Linear-chain CRF layer."""
    
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        
        # Transition parameters: transitions[i, j] = score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        # Initialize with BIO constraints
        self._init_constraints()
    
    def _init_constraints(self):
        """Initialize transition scores with BIO constraints."""
        # Penalize I- tags at the start
        with torch.no_grad():
            for tag, idx in TAG_TO_IDX.items():
                if tag.startswith('I-'):
                    self.start_transitions[idx] = -10000
            
            # Penalize invalid transitions (e.g., O -> I-PER)
            for from_tag, from_idx in TAG_TO_IDX.items():
                for to_tag, to_idx in TAG_TO_IDX.items():
                    if to_tag.startswith('I-'):
                        entity_type = to_tag[2:]
                        # I- must follow B- or I- of same type
                        if not (from_tag == f'B-{entity_type}' or from_tag == f'I-{entity_type}'):
                            self.transitions[to_idx, from_idx] = -10000
    
    def forward(self, emissions, tags, mask):
        """Compute negative log-likelihood loss."""
        # emissions: (batch, seq_len, num_tags)
        # tags: (batch, seq_len)
        # mask: (batch, seq_len)
        
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_algorithm(emissions, mask)
        
        return (forward_score - gold_score).mean()
    
    def _score_sentence(self, emissions, tags, mask):
        """Score the gold tag sequence."""
        batch_size, seq_len, _ = emissions.shape
        
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
        for t in range(1, seq_len):
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t], tags[:, t-1]]
            score += (emit_score + trans_score) * mask[:, t]
        
        # End transition
        last_tags = tags.gather(1, (mask.sum(dim=1) - 1).long().unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        
        return score
    
    def _forward_algorithm(self, emissions, mask):
        """Forward algorithm for partition function."""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Start with start transitions + first emissions
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        
        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(2)  # (batch, num_tags, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            
            # score[b, j] = log(sum_i exp(score[b, i] + trans[j, i] + emit[b, j]))
            next_score = score.unsqueeze(1) + trans_score + emit_score
            next_score = torch.logsumexp(next_score, dim=2)
            
            # Mask: keep old score where mask is 0
            score = next_score * mask[:, t].unsqueeze(1) + score * (1 - mask[:, t].unsqueeze(1))
        
        # Add end transitions
        score += self.end_transitions.unsqueeze(0)
        
        return torch.logsumexp(score, dim=1)
    
    def decode(self, emissions, mask):
        """Viterbi decoding."""
        batch_size, seq_len, num_tags = emissions.shape
        
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []
        
        for t in range(1, seq_len):
            emit_score = emissions[:, t].unsqueeze(2)
            trans_score = self.transitions.unsqueeze(0)
            
            broadcast_score = score.unsqueeze(1) + trans_score + emit_score
            best_score, best_path = broadcast_score.max(dim=2)
            
            history.append(best_path)
            score = best_score * mask[:, t].unsqueeze(1) + score * (1 - mask[:, t].unsqueeze(1))
        
        # Add end transitions
        score += self.end_transitions.unsqueeze(0)
        
        # Backtrack
        best_tags = []
        _, best_last_tag = score.max(dim=1)
        best_tags.append(best_last_tag)
        
        for hist in reversed(history):
            best_last_tag = hist.gather(1, best_last_tag.unsqueeze(1)).squeeze(1)
            best_tags.append(best_last_tag)
        
        best_tags.reverse()
        return torch.stack(best_tags, dim=1)


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for NER."""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 100, hidden_dim: int = 256,
                 num_tags: int = NUM_TAGS, dropout: float = 0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2,
            batch_first=True, bidirectional=True
        )
        
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)
    
    def forward(self, words, tags=None, lengths=None, mask=None):
        """Forward pass."""
        # Embedding
        embeds = self.dropout(self.embedding(words))
        
        # Pack for LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeds)
        
        # Emissions
        emissions = self.hidden2tag(self.dropout(lstm_out))
        
        if mask is None:
            mask = (words != PAD_IDX).float()
        
        if tags is not None:
            # Training: compute loss
            return self.crf(emissions, tags, mask)
        else:
            # Inference: decode
            return self.crf.decode(emissions, mask)


def conll_to_sentences(dataset):
    """Convert HuggingFace CoNLL-2003 to list of (words, tags) tuples."""
    tag_map = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 
               5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    
    sentences = []
    for example in dataset:
        tokens = example['tokens']
        ner_tags = [tag_map.get(t, 'O') for t in example['ner_tags']]
        sentences.append((tokens, ner_tags))
    
    return sentences


def build_vocab(sentences, min_freq: int = 1):
    """Build vocabulary from sentences."""
    word_freq = {}
    for words, _ in sentences:
        for word in words:
            word_lower = word.lower()
            word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
    
    # Special tokens
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    for word, freq in sorted(word_freq.items()):
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    return word_to_idx


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    correct = 0
    predicted = 0
    actual = 0
    
    with torch.no_grad():
        for batch in dataloader:
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['lengths']
            mask = (words != PAD_IDX).float().to(device)
            
            pred_tags = model(words, lengths=lengths, mask=mask)
            
            for i in range(len(lengths)):
                length = lengths[i].item()
                pred = pred_tags[i, :length].cpu().numpy()
                gold = tags[i, :length].cpu().numpy()
                
                for p, g in zip(pred, gold):
                    if g != TAG_TO_IDX['O']:
                        actual += 1
                    if p != TAG_TO_IDX['O']:
                        predicted += 1
                    if g != TAG_TO_IDX['O'] and p == g:
                        correct += 1
    
    precision = correct / predicted if predicted > 0 else 0
    recall = correct / actual if actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def train_bilstm_crf():
    """Train BiLSTM-CRF model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading CoNLL-2003 dataset...")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    
    train_sents = conll_to_sentences(dataset['train'])
    val_sents = conll_to_sentences(dataset['validation'])
    test_sents = conll_to_sentences(dataset['test'])
    
    print(f"Train: {len(train_sents)}, Val: {len(val_sents)}, Test: {len(test_sents)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx = build_vocab(train_sents, min_freq=2)
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Create datasets
    train_dataset = NERDataset(train_sents, word_to_idx)
    val_dataset = NERDataset(val_sents, word_to_idx)
    test_dataset = NERDataset(test_sents, word_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    # Create model
    print("Creating model...")
    model = BiLSTMCRF(
        vocab_size=len(word_to_idx),
        embedding_dim=100,
        hidden_dim=256,
        num_tags=NUM_TAGS,
        dropout=0.5
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Training loop
    print("Training...")
    best_f1 = 0
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            words = batch['words'].to(device)
            tags = batch['tags'].to(device)
            lengths = batch['lengths']
            mask = (words != PAD_IDX).float().to(device)
            
            optimizer.zero_grad()
            loss = model(words, tags=tags, lengths=lengths, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Evaluate
        p, r, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        
        scheduler.step(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'bilstm_crf_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('bilstm_crf_best.pt', weights_only=True))
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    p, r, f1 = evaluate(model, test_loader, device)
    print(f"Test: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()
    model.cpu()
    
    dummy_input = torch.zeros(1, 32, dtype=torch.long)
    dummy_lengths = torch.tensor([32], dtype=torch.long)
    
    # Note: CRF decode is not easily exportable to ONNX
    # We export just the BiLSTM emissions and handle CRF in Rust
    class EmissionModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.embedding = base_model.embedding
            self.lstm = base_model.lstm
            self.hidden2tag = base_model.hidden2tag
        
        def forward(self, words):
            embeds = self.embedding(words)
            lstm_out, _ = self.lstm(embeds)
            return self.hidden2tag(lstm_out)
    
    emission_model = EmissionModel(model)
    
    torch.onnx.export(
        emission_model,
        dummy_input,
        'bilstm_crf_emissions.onnx',
        input_names=['words'],
        output_names=['emissions'],
        dynamic_axes={
            'words': {0: 'batch', 1: 'seq'},
            'emissions': {0: 'batch', 1: 'seq'}
        },
        opset_version=14
    )
    print("Saved emissions model to bilstm_crf_emissions.onnx")
    
    # Export vocabulary and CRF weights
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    crf_weights = {
        'transitions': model.crf.transitions.detach().numpy().tolist(),
        'start_transitions': model.crf.start_transitions.detach().numpy().tolist(),
        'end_transitions': model.crf.end_transitions.detach().numpy().tolist(),
    }
    
    vocab_data = {
        'word_to_idx': word_to_idx,
        'tag_to_idx': TAG_TO_IDX,
        'idx_to_tag': IDX_TO_TAG,
        'crf': crf_weights,
    }
    
    with open('bilstm_crf_vocab.json', 'w') as f:
        json.dump(vocab_data, f, indent=2)
    print("Saved vocabulary to bilstm_crf_vocab.json")
    
    # Also save just CRF weights for Rust fallback
    with open('bilstm_crf_weights.json', 'w') as f:
        json.dump(crf_weights, f, indent=2)
    print("Saved CRF weights to bilstm_crf_weights.json")
    
    return model


if __name__ == '__main__':
    train_bilstm_crf()

