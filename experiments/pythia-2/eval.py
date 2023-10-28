
import random
import time
import os
import sys

import numpy as np
import torch

from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset

model_names = [
    "pythia-19m",
    "pythia-125m",
    "pythia-350m",
    "pythia-800m",
    "pythia-1.3b",
    "pythia-2.7b",
    "pythia-6.7b",
    "pythia-13b"
]

model_names_deduped = [
    "pythia-19m-deduped",
    "pythia-125m-deduped",
    "pythia-350m-deduped",
    "pythia-800m-deduped",
    "pythia-1.3b-deduped",
    "pythia-2.7b-deduped",
    "pythia-6.7b-deduped",
    "pythia-13b-deduped"
]

def tokenize_split(txt, tokenizer):
    """Splits string `txt` according to how `tokenizer` tokenizes it."""
    return tokenizer.batch_decode(tokenizer.encode(txt))

if __name__ == '__main__':
   
    output_dir = "/om/user/ericjm/results/the-everything-machine/pythia-2" 
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
   
    k_idx = int(sys.argv[1]) # index 0...7 where there are 8 models
    step = 143000
    model_name = model_names[k_idx]

    # load the_pile test set
    dataset = load_dataset("json", data_files="/om/user/ericjm/the_pile/test.jsonl.zst", cache_dir="/om/user/ericjm/the_pile/") 
    dataset = dataset['train']

    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{model_name}",
        revision=f"step{step}",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step{step}",
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        f"EleutherAI/{model_names[0]}",
        revision=f"step{step}",
        cache_dir=f"/om/user/ericjm/pythia-models/{model_names[0]}/step{step}",
    )

    results = []
    for i in range(200000):
        prompt = dataset[i]['text']
        if prompt:
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            results.append(ls.tolist())
        else:
            results.append([])
        if i % 2000 == 0:
            print(i)

    torch.save(results, os.path.join(output_dir, f"{model_name}.pt"))

