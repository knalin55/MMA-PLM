"""Model utility helpers for MLM training metrics, embedding construction, and embedding freezing.

This module provides functions to compute token-level accuracy on non-masked positions,
prepare logits for metric computation, load or synthesize embedding matrices from multiple
sources (FastText, model-derived, random), and inject/freeze embeddings in supported
Transformer model architectures.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
import evaluate

# Load the evaluation metric once.
METRIC = evaluate.load("accuracy", device="cpu")
EMBED_DIM = 768
DEFAULT_VOCAB_SIZE = 30000
PATH_TO_EMB = "./data/embeddings"

def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute accuracy over non-masked token positions only."""
    predictions, labels = eval_pred

    valid_indices = [
        [idx for idx, label in enumerate(label_row) if label != -100]
        for label_row in labels
    ]

    flattened_labels = [labels[row][valid_indices[row]] for row in range(len(labels))]
    flattened_labels = [item for sublist in flattened_labels for item in sublist]

    flattened_predictions = [
        predictions[0][row][valid_indices[row]]
        for row in range(len(predictions[0]))
    ]
    flattened_predictions = [item for sublist in flattened_predictions for item in sublist]

    results = METRIC.compute(predictions=flattened_predictions, references=flattened_labels)
    results["eval_accuracy"] = results.pop("accuracy")
    return results


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels


def load_vector(path: str, vocab_size: int = DEFAULT_VOCAB_SIZE, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Load a text-based embedding file into a NumPy matrix."""
    embeddings = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    with open(path, "r", encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i == 0:
                continue
            _, vect = line.rstrip().split(" ", 1)
            embeddings[i - 1] = np.fromstring(vect, sep=" ", dtype=np.float32)
    return embeddings


def create_mbase_embeddings(**kwargs: Any) -> np.ndarray:
    """Create embeddings by averaging mBERT token vector outputs for each token."""
    tokenizer = kwargs["tokenizer"]
    shared_tokenizer = kwargs["shared_tokenizer"]
    base_model = kwargs["base_model"]

    vocab_size = len(tokenizer)
    embeddings = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)

    for token, token_id in tokenizer.get_vocab().items():
        inputs = shared_tokenizer(token, return_tensors="pt")["input_ids"]
        try:
            hidden = base_model.bert.embeddings.word_embeddings(inputs)[0][1:-1]
        except AttributeError:
            hidden = base_model.model.embeddings.tok_embeddings(inputs)[0][1:-1]
        embeddings[token_id] = hidden.mean(dim=0).detach().cpu().numpy()

    return embeddings


def create_random_embeddings(**kwargs: Any) -> np.ndarray:
    """Create a random embedding matrix for the tokenizer vocabulary."""
    tokenizer = kwargs["tokenizer"]
    vocab_size = len(tokenizer)
    return np.random.uniform(low=-1.0, high=1.0, size=(vocab_size, EMBED_DIM)).astype(np.float32)


def create_embeddings(
    emb_type: str,
    vocab: Optional[str],
    lngs: List[str],
    tokenizer: Dict[str, transformers.PreTrainedTokenizerFast],
    shared_tokenizer: transformers.PreTrainedTokenizerFast,
    base_model: transformers.PreTrainedModel,
    avg: bool = False,
    model_name: str = "mbert",
) -> np.ndarray:
    """Construct the final embedding matrix for different vocabulary and embedding strategies."""
    
    if vocab != "model_vocab" and vocab is not None:
        total_size = DEFAULT_VOCAB_SIZE * len(lngs)
        res_emb = np.zeros((total_size, EMBED_DIM), dtype=np.float32)
        base_model_emb = np.zeros((total_size, EMBED_DIM), dtype=np.float32)
    else:
        if model_name == "mmbert":
            res_emb = base_model.model.embeddings.tok_embeddings.weight.detach().cpu().numpy()
        else:
            res_emb = base_model.bert.embeddings.word_embeddings.weight.detach().cpu().numpy()
        base_model_emb = np.zeros_like(res_emb)

    embed_path = defaultdict(dict)
    for lng in lngs:
        embed_path["fasttext"][lng] = [
            np.load,
            f"{PATH_TO_EMB}/fasttext/{lng}/ft_align.npy",
        ]
        embed_path["mbert_ft"][lng] = [
            np.load,
            f"{PATH_TO_EMB}/mbert_ft/embeddings_{lng}.npy",
        ]
        embed_path["model_embed"][lng] = [
            create_mbase_embeddings,
            {"base_model": base_model, "shared_tokenizer": shared_tokenizer, "tokenizer": tokenizer[lng]},
        ]
        embed_path["random"][lng] = [
            create_random_embeddings,
            {"tokenizer": tokenizer[lng]},
        ]

    if vocab == "custom":
        if emb_type in {"mbert", "bert", "mmbert", "xlm", "roberta", "random", "model_embed"}:
            res_emb = np.concatenate(
                [embed_path[emb_type][lng][0](**embed_path[emb_type][lng][1]) for lng in lngs], axis=0
            )
        else:
            print(emb_type)
            try:
                res_emb = np.concatenate(
                    [embed_path[emb_type][lng][0](embed_path[emb_type][lng][1]) for lng in lngs],
                    axis=0,
                )
            except Exception:
                res_emb = np.concatenate(
                    [np.loadtxt(embed_path[emb_type][lng][1]) for lng in lngs], axis=0
                )
    elif vocab == "mix":
        res_emb = embed_path["model_embed"][lngs[0]][0](**embed_path["model_embed"][lngs[0]][1])
    else:
        if emb_type in {"model_embed", "random"}:
            res_emb = np.concatenate(
                [embed_path[emb_type][lng][0](**embed_path[emb_type][lng][1]) for lng in lngs], axis=0
            )
        else:
            res_emb = embed_path[emb_type][lngs[0]]

    for idx, lng in enumerate(lngs):
        for token, token_id in tokenizer[lng].get_vocab().items():
            global_id = len(tokenizer[lng]) * idx + token_id
            if token in shared_tokenizer.all_special_tokens:
                inputs = shared_tokenizer(token, return_tensors="pt")["input_ids"]
                try:
                    hidden = base_model.bert.embeddings.word_embeddings(inputs)[0][1:-1]
                except AttributeError:
                    hidden = base_model.model.embeddings.tok_embeddings(inputs)[0][1:-1]
                res_emb[global_id] = hidden.mean(dim=0).detach().cpu().numpy()

            if avg:
                inputs = shared_tokenizer(token, return_tensors="pt")["input_ids"]
                hidden = base_model.bert.embeddings.word_embeddings(inputs)[0][1:-1]
                base_model_emb[global_id] = hidden.mean(dim=0).detach().cpu().numpy()

        if res_emb.shape[0] == tokenizer[lng].vocab_size:
            break

    if avg:
        res_emb = (res_emb + base_model_emb) / 2.0

    return res_emb


def freeze_emb(
    model: transformers.PreTrainedModel,
    embedding: Optional[np.ndarray] = None,
    train: str = "full",
    model_name: str = "mbert",
) -> transformers.PreTrainedModel:
    """Replace model embeddings and configure trainable parameters."""
    if embedding is None:
        return model

    embedding_tensor = torch.from_numpy(embedding.astype(np.float32))
    vocab_size = embedding.shape[0]

    if model_name == "roberta":
        model.roberta.embeddings.word_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            _weight=embedding_tensor,
        )
        model.config.vocab_size = vocab_size
        model.lm_head.decoder = torch.nn.Linear(in_features=EMBED_DIM, out_features=vocab_size, bias=True)
        model.lm_head.decoder.weight = torch.nn.Parameter(embedding_tensor)
        target_embedding = model.roberta.embeddings.word_embeddings
        target_decoder = model.lm_head.decoder
    elif model_name == "mmbert":
        model.model.embeddings.tok_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            _weight=embedding_tensor,
        )
        model.config.vocab_size = vocab_size
        model.decoder = torch.nn.Linear(in_features=EMBED_DIM, out_features=vocab_size, bias=True)
        model.decoder.weight = torch.nn.Parameter(embedding_tensor)
        target_embedding = model.model.embeddings.tok_embeddings
        target_decoder = model.decoder
    else:
        model.bert.embeddings.word_embeddings = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIM,
            _weight=embedding_tensor,
        )
        model.config.vocab_size = vocab_size
        model.cls.predictions.decoder = torch.nn.Linear(in_features=EMBED_DIM, out_features=vocab_size, bias=True)
        model.cls.predictions.decoder.weight = torch.nn.Parameter(embedding_tensor)
        target_embedding = model.bert.embeddings.word_embeddings
        target_decoder = model.cls.predictions.decoder

    trainable_params = train in {"full", "nonEmb"}
    for param in model.parameters():
        param.requires_grad = trainable_params

    requires_embedding_grad = train in {"full", "embOnly"}
    target_embedding.weight.requires_grad = requires_embedding_grad
    target_decoder.weight.requires_grad = requires_embedding_grad

    return model
