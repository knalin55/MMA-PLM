import argparse
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import transformers

from data_utils import customDataset
from utils import create_embeddings, freeze_emb, compute_metrics, preprocess_logits_for_metrics

CACHE_DIR = "/home/.cache"

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a masked language model across multiple languages.")

    parser.add_argument("--embed_path", default=None, type=str, help="Path to a precomputed embedding file.")
    parser.add_argument("--embed", default=None, type=str, help="Embedding type: model/fasttext/muse/muse_swapped/isovec")
    parser.add_argument("--vocab", default=None, type=str, help="Vocabulary source: custom/mix/model.")
    parser.add_argument("--train", default="full", type=str, help="Training mode: full/nonEmb/embOnly.")
    parser.add_argument("--model", default="google-bert/bert-base-multilingual-cased", type=str, help="Base transformer model to use.")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--lang", default="ga,br,cy", type=str, help="Comma-separated list of languages.")
    parser.add_argument("--resumefromckpt", default=None, type=str, help="Path to resume training from a checkpoint.")
    parser.add_argument("--skip_freeze", default=False, action="store_true", help="Skip embedding freezing and debug training behaviour.")
    parser.add_argument("--lora", default=False, action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--dsize", default=500000, type=int, help="Max number of training examples per language.")
    parser.add_argument("--outputdir", default="./outputs/", type=str, help="Directory to save training outputs.")

    return parser.parse_args()


def select_model(args: argparse.Namespace, cache_dir: Path) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedModel]:
    """Load or initialize the base model and the mBERT model for embedding creation."""
    if args.model == "random":
        config = transformers.AutoConfig.from_pretrained(
            "google-bert/bert-base-multilingual-cased", cache_dir=cache_dir
        )
        model = transformers.AutoModelForMaskedLM.from_config(config)
        base_model = transformers.AutoModelForMaskedLM.from_config(config)
    else:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            args.model, cache_dir=cache_dir, ignore_mismatched_sizes=True
        )
        base_model = transformers.AutoModelForMaskedLM.from_pretrained(
            args.model, cache_dir=cache_dir, ignore_mismatched_sizes=True
        )
    return model, base_model


def main(args: argparse.Namespace) -> None:
    """Main training pipeline"""

    transformers.set_seed(args.seed)

    # Define dataset and cache paths
    data_root = "/home/nkumar/personal_work_troja/nkumar/phd_thesis/experiments/plugnplay/experiments/data/mono"
    cache_dir = Path(CACHE_DIR)

    # Load and tokenize datasets for all languages
    dataset_loader = customDataset(args.lang, args, cache_dir=str(cache_dir))
    dataset_train, dataset_val, tokenizers = dataset_loader.forward(data_root)
    shared_tokenizer = dataset_loader._load_shared_tokenizer()
    
    # Use the first language tokenizer for data collation
    langs = [lng.strip() for lng in args.lang.split(",") if lng.strip()]
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizers[langs[0]],
        mlm=True,
        mlm_probability=0.2,
    )

    # Load models for training and embedding creation
    model, base_model = select_model(args, cache_dir)

    embed = None
    model_name = "mbert" if re.search("multilingual|mbert", args.model) else args.model.split("/")[-1].split("-")[0].lower()

    if args.embed_path:
        embed = np.load(args.embed_path)
    else:
        embed_type = "model_embed" if args.embed in {"mbert", "bert", "mmbert", "xlm", "roberta", "random", "model"} else args.embed
        vocab_type = "model_vocab" if args.vocab in {"mbert", "bert", "xlm", "roberta", "mmbert", "model"} else args.vocab
        if args.embed is not None or args.vocab is not None:
            embed = create_embeddings(
                embed_type,
                vocab_type,
                langs,
                tokenizers,
                shared_tokenizer,
                base_model,
                avg=False,
                model_name=model_name,
            )

    if not args.skip_freeze:
        model = freeze_emb(model, embed, args.train, model_name=model_name)

    # Prepare LoRA if requested.
    if args.lora:
        from peft import LoraConfig, LoraModel

        lora_config = LoraConfig(
            task_type="TOKEN_CLS",
            r=64,
            lora_alpha=128,
            target_modules=["all-linear"],
            lora_dropout=0.01,
        )
        model = LoraModel(model, lora_config, "default")

    ln = args.lang.replace(",", "_")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(
        f"{args.outputdir}/{ln}/{model_name}/{args.embed}/{args.vocab}/{args.dsize}/{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        label_names=["labels"],
        save_safetensors=False,
        remove_unused_columns=True,
    )

    model_path_dir = Path(f"./{ln}")
    model_path_dir.mkdir(exist_ok=True)
    with (model_path_dir / "model_path").open("w", encoding="utf-8") as path_file:
        print(output_dir, file=path_file)
        print(sys.argv[1:], file=path_file)

    device = "cuda"
    model = model.to(device)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train(args.resumefromckpt)

    if args.lora:
        model_res = trainer.model.merge_and_unload()
        save_dir = output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        model_res.save_pretrained(save_dir / f"best_{args.seed}")
    else:
        trainer.save_model(output_dir / f"best_{args.seed}")

    # Remove temporary checkpoint folders and move the final output directory.
    for child in output_dir.iterdir():
        if child.name.startswith("check"):
            shutil.rmtree(child)



if __name__ == "__main__":
    main(parse_args())
