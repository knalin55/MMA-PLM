"""Utilities for dataset loading and tokenizer creation."""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
import transformers
from datasets import Dataset

def preprocess_batch(
    examples: dict, tokenizer_: transformers.PreTrainedTokenizerFast, shift_index: int
) -> dict:
    """Tokenize text examples and optionally shift input IDs for combining custom vocabularies."""
    texts = examples["text"]
    model_inputs = tokenizer_(texts, truncation=True, max_length=512)
    model_inputs["input_ids"] = [
        [token_id + tokenizer_.vocab_size * shift_index for token_id in token_ids]
        for token_ids in model_inputs["input_ids"]
    ]
    return model_inputs


class customDataset():
    """Orchestrates multilingual dataset loading, tokenization, and preprocessing for masked language modeling."""

    def __init__(self, langs: str, args: argparse.Namespace, cache_dir: str = "/home/nkumar/personal_work_ms/nkumar/.cache"):
        """
        Initialize customDataset with language list and configuration.

        Args:
            langs: Comma-separated string of language codes (e.g., "en,fr,de").
            args: Argument namespace containing model, vocab, and training configuration.
            cache_dir: Path to cache directory for pretrained models.
        """
        self.langs = [lng.strip() for lng in langs.split(",") if lng.strip()]
        self.args = args
        self.cache_dir = cache_dir
        self.shared_tokenizer = self._load_shared_tokenizer()

    def _load_shared_tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        """Load the shared tokenizer based on model selection."""
        if self.args.model == "random":
            return transformers.AutoTokenizer.from_pretrained(
                "google-bert/bert-base-multilingual-cased", cache_dir=self.cache_dir
            )
        else:
            return transformers.AutoTokenizer.from_pretrained(self.args.model, cache_dir=self.cache_dir)

    def _load_language_examples(self, data_root: str, lang: str, split: str, max_examples: int) -> dict:
        """
        Load text examples for a given language and dataset split.

        Args:
            data_root: Root directory containing language subdirectories.
            lang: Language code.
            split: Dataset split ("train", "val", or "test").
            max_examples: Maximum number of examples to load.

        Returns:
            Dictionary with "text" and "lang" lists.
        """
        file_path = os.path.join(data_root, lang, split)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing data file: {file_path}")

        texts = []
        langs = []
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if len(line.split()) > 20 and len(texts) < max_examples:
                    texts.append(line.strip())
                    langs.append(lang)
            infile.close()

        return {"text": texts, "lang": langs}

    def _build_raw_datasets(self, data_root: str, max_train_examples: int) -> dict:
        """
        Build Hugging Face Dataset objects for all languages and splits.

        Args:
            data_root: Root directory containing language subdirectories.
            max_train_examples: Maximum number of training examples per language.

        Returns:
            Nested dictionary with structure: {lang: {split: Dataset}}.
        """
        datasets = defaultdict(dict)
        for lang in self.langs:
            for split in ["train", "val", "test"]:
                max_examples = max_train_examples if split == "train" else 5000
                examples = self._load_language_examples(data_root, lang, split, max_examples)
                datasets[lang][split] = Dataset.from_dict(examples)
        return datasets

    def _get_tokenizer_for_language(self, lang: str) -> transformers.PreTrainedTokenizerFast:
        """
        Get the appropriate tokenizer for a language based on vocabulary strategy.

        Args:
            lang: Language code.

        Returns:
            Configured tokenizer instance.
        """
        if self.args.vocab == "custom":
            tokenizer = transformers.PreTrainedTokenizerFast(
                tokenizer_file=f"/home/nkumar/personal_work_troja/nkumar/phd_thesis/experiments/plugnplay/experiments/data/tokens/{lang}.json"
            )
        elif self.args.vocab == "mix":
            tokenizer = transformers.PreTrainedTokenizerFast(
                tokenizer_file=f"/home/nkumar/personal_work_troja/nkumar/phd_thesis/experiments/plugnplay/experiments/data/tokens/br_ga_cy.json"
            )
        else:
            tokenizer = self.shared_tokenizer

        # Set special tokens for custom/mix vocabularies
        if self.args.vocab == "custom" or self.args.vocab == "mix":
            tokenizer.pad_token = "[PAD]"
            tokenizer.eos_token = "[PAD]"
            tokenizer.sep_token = "[SEP]"
            tokenizer.unk_token = "[UNK]"
            tokenizer.cls_token = "[CLS]"
            tokenizer.mask_token = "[MASK]"
            tokenizer.model_max_length = 512

        return tokenizer

    def _tokenize_batch(
        self, examples: dict, tokenizer_: transformers.PreTrainedTokenizerFast, shift_index: int
    ) -> dict:
        """
        Tokenize a batch of text examples with optional vocabulary ID shifting.

        Args:
            examples: Dictionary with "text" field containing text examples.
            tokenizer_: Tokenizer instance to use.
            shift_index: Index for shifting token IDs (used to separate vocabularies).

        Returns:
            Dictionary with tokenized input_ids, token_type_ids, and attention_mask.
        """
        texts = examples["text"]
        model_inputs = tokenizer_(texts, truncation=True, max_length=512)

        # Shift token IDs if using custom vocabularies to keep vocabulary ranges separate

        model_inputs["input_ids"] = [
            [token_id + tokenizer_.vocab_size * shift_index for token_id in token_ids]
            for token_ids in model_inputs["input_ids"]
        ]

        return model_inputs

    def forward(self, data_root: str):
        """
        Load all language datasets, tokenize them, and combine into unified train/val datasets.

        Args:
            data_root: Root directory containing language subdirectories.

        Returns:
            Tuple of (combined_train_dataset, combined_val_dataset, per_language_tokenizers).
        """
        # Load raw datasets for each language
        raw_datasets = self._build_raw_datasets(data_root, self.args.dsize)

        # Create language-specific tokenizers
        tokenizers = {lang: self._get_tokenizer_for_language(lang) for lang in self.langs}

        # Tokenize each language dataset
        tokenized_train = {}
        tokenized_val = {}
        for index, lang in enumerate(self.langs):
            shift_index = index if self.args.vocab == "custom" else 0
            tokenized_train[lang] = raw_datasets[lang]["train"].map(
                self._tokenize_batch,
                batched=True,
                num_proc=1,
                fn_kwargs={"tokenizer_": tokenizers[lang], "shift_index": shift_index},
            )
            tokenized_val[lang] = raw_datasets[lang]["val"].map(
                self._tokenize_batch,
                batched=True,
                num_proc=1,
                fn_kwargs={"tokenizer_": tokenizers[lang], "shift_index": shift_index},
            )

        # Combine tokenized datasets across languages
        train_df = pd.concat([pd.DataFrame(tokenized_train[lang]) for lang in self.langs], ignore_index=True)
        val_df = pd.concat([pd.DataFrame(tokenized_val[lang]) for lang in self.langs], ignore_index=True)

        dataset_train = Dataset.from_pandas(train_df, preserve_index=False)
        dataset_val = Dataset.from_pandas(val_df, preserve_index=False)

        # Shuffle and remove auxiliary columns
        dataset_train = dataset_train.shuffle(seed=self.args.seed).remove_columns(["text", "lang"])
        dataset_val = dataset_val.shuffle(seed=self.args.seed).remove_columns(["text", "lang"])

        return dataset_train, dataset_val, tokenizers
