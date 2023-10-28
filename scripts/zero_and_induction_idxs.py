#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import scipy.linalg
import torch
import torch.nn.functional as F
import sklearn.cluster

import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM


pile_canonical = "/om/user/ericjm/the_pile/the_pile_test_canonical_200k"
# ----- load the_pile test set -----
dataset = datasets.load_from_disk(pile_canonical)

starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))

def loss_idx_to_dataset_idx(idx):
    """given an idx in range(0, 10658635), return
    a sample index in range(0, 20000) and pred-in-sample
    index in range(0, 1023). Note token-in-sample idx is
    exactly pred-in-sample + 1"""
    sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
    pred_in_sample_index = idx - starting_indexes[sample_index]
    return int(sample_index), int(pred_in_sample_index)

def get_context(idx):
    """given idx in range(0, 10658635), return dataset sample
    and predicted token index within sample, in range(1, 1024)."""
    sample_index, pred_index = loss_idx_to_dataset_idx(idx)
    return dataset[sample_index], pred_index+1

def print_context(idx, context_length=-1):
    """
    given idx in range(0, 10658635), print prompt preceding the corresponding
    prediction, and highlight the predicted token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
    if context_length > 0:
        prompt = prompt[-context_length:]
    prompt = "".join(prompt)
    token = sample["split_by_token"][token_idx]
    print(prompt + "\033[41m" + token + "\033[0m")

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

curves = np.load("/om/user/ericjm/results/the-everything-machine/pythia-2/pythia-2.npy")

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

model_name = model_names[0]
step = 143000

# model = GPTNeoXForCausalLM.from_pretrained(
#     f"EleutherAI/{model_name}",
#     revision=f"step{step}",
#     cache_dir=f"/om/user/ericjm/pythia-models/{model_name}/step{step}",
# ).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/{model_names[0]}",
    revision=f"step{step}",
    cache_dir=f"/om/user/ericjm/pythia-models/{model_names[0]}/step{step}",
)

zero_idxs = curves[:, 0] < 0.1
zero_idxs, = zero_idxs.nonzero()

# induction_idxs = np.zeros((curves.shape[0],), dtype='bool')
induction_idxs = []
i = 0
for document_idx in tqdm(range(len(dataset))):
    document = dataset[document_idx]
    document_trigrams = defaultdict(int)
    tokens = document['input_ids'][0]
    if len(tokens) > 1:
        i += 1
        for j in range(2, len(tokens)):
            trigram = tuple(tokens[j-2:j+1])
            if trigram in document_trigrams:
                # induction_idxs[i] = 1
                induction_idxs.append(i)
            document_trigrams[trigram] += 1
            i += 1

non_induction_zeros = set(zero_idxs).difference(set(induction_idxs))
non_induction_zeros = sorted(list(non_induction_zeros))

with open("../zero_and_induction_idxs.pkl", "wb") as f:
    pickle.dump((non_induction_zeros, zero_idxs, induction_idxs), f)

