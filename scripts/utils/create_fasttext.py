"""Train FastText-based token embeddings for a language and export aligned embedding artifacts.

This script builds token embeddings from monolingual corpora using FastText, initializes
special-token vectors from multilingual BERT, and saves both initial and alignment-refined
embedding matrices for downstream MLM experiments.
"""

from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer, BertModel
import torch
import os
from gensim.models import FastText
import numpy as np
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, help="Language code -- e.g., 'gd', 'cy', 'ga', etc.")
parser.add_argument("--inputdir", type=str, default="./data/mono", help="Directory containing the input monolingual text files")
parser.add_argument("--outputdir", type=str, default="./data/embeddings", help="Directory where embeddings are going to be saved")
parser.add_argument("--tokenizer", type=str, default="custom", help="Tokenizer to use: 'custom' or the multilingual BERT tokenizer") # To-Do: Add option for other models like XLM-RoBERTa, etc.
parser.add_argument("--tokenizer_path", type=str, default="./data/tokens", help="Directory containing custom tokenizer JSON files")
parser.add_argument("--cache_dir", type=str, default="./cache", help="Directory used to cache pretrained model downloads")
args = parser.parse_args()


PATH_TO_MONO = args.inputdir
PATH_TO_FT_EMBED = args.outputdir
PATH_TO_TOKEN = args.tokenizer_path
CACHE_DIR = args.cache_dir

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def main(args: argparse.Namespace) -> None:

    lang = args.language

    mtokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased", cache_dir=CACHE_DIR)
    mbert = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased", cache_dir=CACHE_DIR)
    
    if args.tokenizer == "custom":
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{PATH_TO_TOKEN}/{lang}.json")
    else:
        tokenizer = mtokenizer

    train = []
    val = []
    test = []

    with open(f"{PATH_TO_MONO}/{lang}/train") as file:
        for line in file:
            train.append(line)
        file.close()
    with open(f"{PATH_TO_MONO}/{lang}/val") as file:
        for line in file:
            val.append(line)
        file.close()
    with open(f"{PATH_TO_MONO}/{lang}/test") as file:
        for line in file:
            test.append(line)
        file.close()

    tokenized_inputs = tokenizer(train+val+test)
    corpus = [tokenized_inputs.tokens(i) for i in range(len(train+val+test))]

    print("--- Training FastText model ---")
    model = FastText(vector_size=768, window=5, min_count=1, workers=4, sg=1)
    model.build_vocab(corpus)
    model.train(corpus,total_examples=len(corpus),epochs=1)

    if args.tokenizer == "custom":
        res_emb = np.zeros((30000, 768))
    else:
        res_emb = mbert.embeddings.word_embeddings.weight.to(device="cpu").detach().numpy()
    
    for token, id in tokenizer.get_vocab().items():
        if token in mtokenizer.all_special_tokens:
            res_emb[id] = mbert.embeddings.word_embeddings(mtokenizer(token, return_tensors="pt")['input_ids'])[0][1:-1].mean(axis=0).detach().numpy()
        else:
            try: 
                res_emb[id] = model.wv[token]
            except:
                pass

    if not os.path.exists(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}"):
        os.makedirs(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}")

    fname = get_tmpfile(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}/fasttext.model")
    # model.save(fname)

    outfile = open(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}/ft.npy", "wb")
    np.save(outfile, res_emb)

    print("--- Retraining FastText model with better alignment ---")

    mbert_vectors = []
    for key, value in model.wv.key_to_index.items():
        mbert_vectors.append(mbert.embeddings.word_embeddings(mtokenizer(key, return_tensors="pt")['input_ids'])[0][1:-1].mean(axis=0).detach().numpy())
    

    # Retrain FastText model with mBERT initialization
    model_align = FastText(vector_size=768, window=5, min_count=1, workers=4, sg=1)
    model_align.build_vocab(corpus)

    model_align.wv.vectors = np.array(mbert_vectors)
    model_align.train(corpus,total_examples=len(corpus),epochs=5, compute_loss=True, callbacks=[callback()])

    if args.tokenizer == "custom":
        res_emb = np.zeros((30000, 768))
    else:
        res_emb = mbert.embeddings.word_embeddings.weight.to(device="cpu").detach().numpy()
    
    for token, id in tokenizer.get_vocab().items():
        if token in mtokenizer.all_special_tokens:
            res_emb[id] = mbert.embeddings.word_embeddings(mtokenizer(token, return_tensors="pt")['input_ids'])[0][1:-1].mean(axis=0).detach().numpy()
        else:
            try: 
                res_emb[id] = model_align.wv[token]
            except:
                pass

    outfile = open(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}/ft_align.npy", "wb")
    np.save(outfile, res_emb)
    fname = get_tmpfile(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}/fasttext_align.model")
    model_align.save(fname)
    import json

    outfile = open(f"{PATH_TO_FT_EMBED}/{lang}/{args.tokenizer}/tokenmap.json", "w")
    json.dump(tokenizer.get_vocab(), fp=outfile, indent=6)
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
