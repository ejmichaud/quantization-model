
import os
import argparse

import numpy as np
from tqdm.auto import tqdm

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':

    # ----- define command line arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str,
                        default="/om/user/ericjm/quanta-discovery/cache/",
                        help="directory of models, tokenizers, losses, etc.")
    parser.add_argument("--directory_containing_zst", type=str,
                        help="directory that test.jsonl.zst is in")
    parser.add_argument("--pile_canonical", type=str,
                    default="/om/user/ericjm/the_pile/the_pile_test_canonical_200k",
                    help="path to save canonical Pile test set")

    args = parser.parse_args()
    model_name = "pythia-70m"     # for tokenizer, note that tokenizers are the same across Pythia models
    step = 143000                 # for tokenizer, note that tokenizers are the same across Pythia models
    num_load_documents = 200000   # we'll tokenize 200k documents
    cache_dir = args.cache_dir
    pile_cache_dir = args.directory_containing_zst
    pile_canonical = args.pile_canonical

    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=os.path.join(cache_dir, model_name, f"step{step}"),
    )

    # ----- load the_pile test set -----
    dataset = load_dataset("json", data_files=os.path.join(pile_cache_dir, "test.jsonl.zst"),
                            cache_dir=pile_cache_dir, split=f"train[:{num_load_documents}]") 

    def tokenize_sample(sample):
        tokens = tokenizer(sample["text"], return_tensors='pt', 
                            max_length=1024, truncation=True)["input_ids"]
        return {"input_ids": tokens}

    dataset = dataset.map(tokenize_sample, load_from_cache_file=True)
    dataset = dataset.map(lambda sample: {"split_by_token": tokenizer.batch_decode(sample["input_ids"][0])}, load_from_cache_file=True)
    dataset = dataset.map(lambda sample: {"tokens_len": len(sample["input_ids"][0])}, load_from_cache_file=True)
    dataset = dataset.map(lambda sample: {"preds_len": max(sample["tokens_len"] - 1, 0)}, load_from_cache_file=True) # fixed this on 2023-02-06 to accomodate empty documents
    starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

    # save dataset
    dataset.save_to_disk(pile_canonical)
