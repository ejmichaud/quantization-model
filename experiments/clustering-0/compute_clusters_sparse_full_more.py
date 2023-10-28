from collections import defaultdict
import os

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import scipy.linalg
import torch
import torch.nn.functional as F
import sklearn.cluster

from datasets import load_dataset
from transformers import AutoTokenizer

import pickle

# load the_pile test set
# dataset = load_dataset("json", data_files="/om/user/ericjm/the_pile/test.jsonl.zst", cache_dir="/om/user/ericjm/the_pile/", split="train[:200000]") 
# tokenizer = AutoTokenizer.from_pretrained(
#     f"EleutherAI/pythia-19m",
#     revision=f"step143000",
#     cache_dir=f"/om/user/ericjm/pythia-models/pythia-19m/step143000",
# )

# def tokenize_sample(sample):
#     tokens = tokenizer(sample["text"], return_tensors='pt', max_length=1024, truncation=True)["input_ids"]
#     return {"input_ids": tokens}


# dataset = dataset.map(tokenize_sample)
# dataset = dataset.map(lambda sample: {"split_by_token": tokenizer.batch_decode(sample["input_ids"][0])})
# dataset = dataset.map(lambda sample: {"tokens_len": len(sample["input_ids"][0])})
# dataset = dataset.map(lambda sample: {"preds_len": max(sample["tokens_len"] - 1, 0)}) # fixed this on 2023-02-06 to accomodate empty documents
# starting_indexes = np.array([0] + list(np.cumsum(dataset["preds_len"])))


# def loss_idx_to_dataset_idx(idx):
#     """given an idx in range(0, 10658635), return
#     a sample index in range(0, 20000) and pred-in-sample
#     index in range(0, 1023). Note token-in-sample idx is
#     exactly pred-in-sample + 1"""
#     sample_index = np.searchsorted(starting_indexes, idx, side="right") - 1
#     pred_in_sample_index = idx - starting_indexes[sample_index]
#     return int(sample_index), int(pred_in_sample_index)


# def get_context(idx):
#     """given idx in range(0, 10658635), return dataset sample
#     and predicted token index within sample, in range(1, 1024)."""
#     sample_index, pred_index = loss_idx_to_dataset_idx(idx)
#     return dataset[sample_index], pred_index+1


# def print_context(idx):
#     """
#     given idx in range(0, 10658635), print prompt preceding the corresponding
#     prediction, and highlight the predicted token.
#     """
#     sample, token_idx = get_context(idx)
#     prompt = sample["split_by_token"][:token_idx]
#     prompt = "".join(prompt)
#     token = sample["split_by_token"][token_idx]
#     print(prompt + "\033[41m" + token + "\033[0m")


# import pickle
# with open("../../tmp/zero_and_induction_idxs.pkl", "rb") as f:
#     non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)

# print(len(zero_idxs))
# print(len(induction_idxs))
# print(len(non_induction_zeros))

idxs, C, C_abs = torch.load("/om/user/ericjm/results/the-everything-machine/clustering-0/full_more.pt")
C = C.cpu().numpy()
C_abs = C_abs.cpu().numpy()
C = 1 - np.arccos(C) / np.pi

# plt.imshow(C)
# plt.xlim(0, 100)
# plt.ylim(100, 0)
# # check if matrix C has any NaNs
# np.isnan(C).any()
# np.isinf(C).any()

CLUSTER_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 
                100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400,
                 500, 600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500]

results = dict()

for n_clusters in tqdm(CLUSTER_COUNTS):
    clusters_labels = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30).fit_predict(C)
    clusters_labels_abs = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=30).fit_predict(C_abs)
    results[n_clusters] = (clusters_labels.tolist(), clusters_labels_abs.tolist())

    with open("/om/user/ericjm/results/the-everything-machine/clustering-0/clusters_full_more.pkl", "wb") as f:
        pickle.dump(results, f)


#     label_frequencies = defaultdict(int)
#     for l in clusters_labels:
#         label_frequencies[l] += 1

#     labels_sorted_by_freq = sorted(label_frequencies.keys(), key=lambda k: label_frequencies[k], reverse=True)
# # label_permutation = [labels_sorted_by_freq.index(i) for i in labels_sorted_by_freq]
# permutation = []
# indices = defaultdict(list)
# for i, cls in enumerate(clusters_labels):
#     indices[cls].append(i)
# for cls in labels_sorted_by_freq:
#     permutation.extend(indices[cls])



