# MMA-PLM: Modular Monolingual Adaptation using Pretrained Language Models

The repository contains the implementation of MMA-PLM to improve the performance of monolingual LMs for low-resource languages.
Don't forget to cite our paper if you use our work. 

```
@inproceedings{kumar-dusek-2026-modular,
    title = "Modular Monolingual Adaptation using Pretrained Language Models",
    author = "Kumar, Nalin  and
      Dusek, Ondrej",
    editor = "Li, Yunyao  and
      Rehm, Georg  and
      Tu, Mei",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 6: Industry Track)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-industry.125/",
    doi = "10.18653/v1/2026.acl-industry.125",
    pages = "1819--1828",
    ISBN = "979-8-89176-394-4",
    abstract = "Building monolingual language models (LMs) for low-resource languages typically relies on adapting pretrained language models (PLMs) by finetuning the whole model on the target language. This approach is widely favored over training from scratch, as it enables effective knowledge transfer. Additionally, prior work has shown that using a language-specific tokenizer can enhance the adaptability. In this work, we hypothesize that full model tuning is often unnecessary and propose a more modular approach. Specifically, we replace the tokens, freeze the corresponding embeddings, and tune the rest of the model. We use Scottish Gaelic, Irish, and Quechua for our experiments, with Quechua being a very low-resource language (8.5k training instances). Evaluation on natural language understanding (NLU) tasks {--} mask-filling, NER, and POS {--} shows that our proposed approach improves performance when adapting the models to low-resource languages. Additionally, we provide a comprehensive analysis of the effectiveness of training strategies, the choice of pretrained embeddings, and models."
}
```
[Link to the paper](https://aclanthology.org/2026.acl-industry.125/)

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/knalin55/MMA-PLM.git
   cd MMA-PLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Linux/macOS
   # .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Requirements

- Python 3.10+ (recommended)
- Torch
- Transformers
- peft, if LoRA is used

Detailed versions are available in `requirements.txt`. 

---

## Download data

Download CC-100 data from https://data.statmt.org/cc-100/

Split it into train, test and val sets. 

Expected structure (example):
```text
data/
  mono/
    {lng}/
        train
        test
        val
```

---

## Create tokenizer

Train a tokenizer from your monolingual corpus. Save the pretrained tokenizers in `data/tokens/{lng}`

Example usage: 
```bash
python scripts/create_tokenizer.py \
  --input_dir data/mono/ \
  --vocab_size 30000 \
  --output_dir data/tokens
```

## Training embeddings

If your pipeline includes a FastText embeddings, train the embeddings using:

```bash
python scripts/train_embeddings.py \
  --language {lng}
  --inputdir ./data/mono
  --outputdir ./data/embeddings
  --tokenizer custom
  --tokenizer_path ./data/tokens
```

--- 

## Training the Language Model

Run base model training:

```bash
python train.py 
```

Typical configurable items:
- input dataset path
- language(s)
- vocabulary to use : model/custom
- embeddings to use : model/fasttext
- max steps / epochs

Example with overrides:
```bash
python train.py \
  --output_dir ./outputs \
  --batch_size 64 \
  --lr 2e-5 \
  --lang gd
  --vocab custom
  --embed model
```

---

## Running LoRA

Fine-tune with LoRA adapters:

```bash
python train.py --lora
```

---
