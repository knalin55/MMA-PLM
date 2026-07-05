This directory contains utility scripts and helper modules for tokenizer creation, embedding generation, dataset preprocessing, and layer freezing modules used in the experiments.

## Files

### `create_tokenizer.py`
Trains and saves a custom WordPiece tokenizer from one or more language corpora.

**Key features**
- Supports single language (`--language`) and multi-language (`--languages`) training.
- Uses `bert-base-multilingual-cased` special tokens for better standardization.
- Saves tokenizer JSON to `./data/tokens` by default.

**Example usage**
```bash
python scripts/utils/create_tokenizer.py \
  --languages ga cy \
  --vocab_size 30000 \
  --inputdir ./data/mono \
  --outputdir ./data/tokens
```

---

### `create_fasttext.py`
Builds FastText token embeddings for a language and exports:
- initial embeddings (`ft.npy`)
- FastText embeddings initialized with mBERT embedding weights (`ft_align.npy`)

**Example**
```bash
python scripts/utils/create_fasttext.py \
  --language ga \
  --inputdir ./data/mono \
  --outputdir ./data/embeddings \
  --tokenizer custom \
  --tokenizer_path ./data/tokens \
  --cache_dir ./cache
```

---

### `data_utils.py`
Provides dataset loading and tokenization utilities for masked language model training.

**Main components**
- `customDataset`: end-to-end dataset loader and tokenizer module.
- Split loading (`train`, `val`, `test`) per language.
- Custom/mixed/model vocab tokenizer selection.

**Expected data layout**
```text
data/mono/<lang>/train
data/mono/<lang>/val
data/mono/<lang>/test
```

---

### `model_utils.py`
Utilities for MLM metrics, embedding creation, and embedding injection/freezing.

**Main functionality**
- `compute_metrics`: token-level accuracy on non-masked positions.
- `preprocess_logits_for_metrics`: memory-efficient logits preprocessing.
- `create_embeddings`: builds embeddings from sources such as:
  - FastText-aligned vectors
  - model-initialized vectors
  - random initialization
- `freeze_emb`: replaces model embeddings and configures trainability for:
  - BERT-style models
  - RoBERTa
  - mM-BERT-like architectures

---
