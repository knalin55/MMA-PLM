"""Train and save custom WordPiece tokenizers from one or more language corpora.

This script builds a BERT-style WordPiece tokenizer using train/val/test files for the
provided languages, applies standard special-token post-processing, and writes a tokenizer
JSON artifact for downstream training pipelines.
"""

from transformers import AutoTokenizer
import os
import random
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from tokenizers.models import WordPiece

from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

CACHE_DIR = "./cache"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--languages",
    nargs="+",
    help="Space-separated list of input wiki languages to train a shared tokenizer on",
    default=None,
)
parser.add_argument(
    "--language", 
    type=str, 
    help="Single input wiki language (deprecated, use --languages)"
)
parser.add_argument(
    "--vocab_size", 
    type=int, 
    default=30000, 
    help="Vocabulary size for the tokenizer"
)
parser.add_argument(
    "--inputdir",
    type=str,
    default="./data/mono",
    help="Root directory containing language subdirectories",
)
parser.add_argument(
    "--outputdir",
    type=str,
    default="./data/tokens",
    help="Directory where the tokenizer JSON will be written",
)

def gather_language_files(inputdir: str, languages):
    files = []
    for language in languages:
        language_dir = os.path.join(inputdir, language)
        for split in ["train", "val", "test"]:
            path = os.path.join(language_dir, split)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected corpus file not found: {path}")
            files.append(path)
    return files


def main(args: argparse.Namespace) -> None:

    tokenizer = Tokenizer(WordPiece())
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    mtokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", cache_dir=CACHE_DIR)
    trainer = WordPieceTrainer(special_tokens=list(mtokenizer.special_tokens_map.values()), vocab_size=args.vocab_size)

    if args.languages is None and args.language is None:
        raise ValueError("Please pass at least one language via --languages or --language")

    languages = args.languages if args.languages is not None else [args.language]
    tokenizer_files = gather_language_files(args.inputdir, languages)

    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[PAD]"
    tokenizer.sep_token = "[SEP]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.model_max_length = 512

    tokenizer.train(files=tokenizer_files, trainer=trainer)
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    os.makedirs(args.outputdir, exist_ok=True)
    output_name = "_".join(languages)
    tokenizer.save(os.path.join(args.outputdir, f"{output_name}.json"))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
