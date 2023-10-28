


from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import scipy.linalg
import torch
import torch.nn.functional as F
import sklearn.cluster

from datasets import load_dataset, set_caching_enabled
from transformers import AutoTokenizer, GPTNeoXForCausalLM

set_caching_enabled(True)

# load the_pile test set
dataset = load_dataset("json", data_files="/om/user/ericjm/the_pile/test.jsonl.zst", cache_dir="/om/user/ericjm/the_pile/", split="train[:200000]") 
tokenizer = AutoTokenizer.from_pretrained(
    f"EleutherAI/pythia-19m",
    revision=f"step143000",
    cache_dir=f"/om/user/ericjm/pythia-models/pythia-19m/step143000",
)

def tokenize_sample(sample):
    tokens = tokenizer(sample["text"], return_tensors='pt', max_length=1024, truncation=True)["input_ids"]
    return {"input_ids": tokens}


dataset = dataset.map(tokenize_sample, load_from_cache_file=True)
dataset = dataset.map(lambda sample: {"split_by_token": tokenizer.batch_decode(sample["input_ids"][0])}, load_from_cache_file=True)
dataset = dataset.map(lambda sample: {"tokens_len": len(sample["input_ids"][0])}, load_from_cache_file=True)
dataset = dataset.map(lambda sample: {"preds_len": max(sample["tokens_len"] - 1, 0)}, load_from_cache_file=True) # fixed this on 2023-02-06 to accomodate empty documents
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


def print_context(idx):
    """
    given idx in range(0, 10658635), print prompt preceding the corresponding
    prediction, and highlight the predicted token.
    """
    sample, token_idx = get_context(idx)
    prompt = sample["split_by_token"][:token_idx]
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
sizes = torch.load("/om/user/ericjm/results/the-everything-machine/pythia-0/num_params.pt")
no_emb_params = [sizes[mn][0] for mn in model_names[:-1]]
with_emb_params = [sizes[mn][2] for mn in model_names[:-1]]
sizes
### Load up model
device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
device
model_name = model_names[0]
step = 143000

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

import pickle
with open("../../tmp/zero_and_induction_idxs.pkl", "rb") as f:
    non_induction_zeros, zero_idxs, induction_idxs = pickle.load(f)
print(len(zero_idxs))
print(len(induction_idxs))
print(len(non_induction_zeros))


# context_and_token = []
# for i in tqdm(non_induction_zeros[:100000]):
#     sample, token_idx = get_context(i)
#     context = sample['input_ids'][0][:token_idx]
#     token = sample['input_ids'][0][token_idx]
#     context_and_token.append((i, context, token))
# hist = defaultdict(list)
# for i, context, token in tqdm(context_and_token):
#     if len(context) > 2:
#         hist[(context[-3], context[-2], context[-1], token)].append(i)
# fourgrams_sorted_by_freq = sorted(list(hist.keys()), key=lambda k: len(hist[k]), reverse=True)



def get_flattened_gradient(model, param_subset):
    grads = []
    for name, p in model.named_parameters():
        if name in param_subset:
            grads.append(p.grad)
    return torch.cat([g.flatten() for g in grads])
param_names = [n for n, p in model.named_parameters()]

highsignal_names = [name for name in param_names if 
                        ('layernorm' not in name) and 
                        ('embed' not in name)]

len_g = sum(model.state_dict()[name].numel() for name in highsignal_names)
idxs = non_induction_zeros[::20][:20000]
# idxs = sum([hist[quad] for quad in fourgrams_sorted_by_freq[8:14]], start=[])
# idxs = induction_idxs[::100][:50] + hist[fourgrams_sorted_by_freq[5]][:40] + hist[fourgrams_sorted_by_freq[10]][:40] + hist[fourgrams_sorted_by_freq[50]][:20]
# random_idxs = [idx for idx in non_induction_zeros[::500] if idx not in idxs]
# idxs = idxs + random_idxs[:50]
S = len(idxs)

print(S)

block_len = 150
blocks = [idxs[i:min(len(idxs), i+block_len)] for i in range(0, len(idxs), block_len)]

C = torch.zeros((S, S), device=device)
C_abs = torch.zeros((S, S), device=device)
iouter = 0
for iblock in tqdm(blocks):
    Gi = torch.zeros((len(iblock), len_g), device=device)
    for i, idx in enumerate(iblock):
        model.zero_grad()
        document, l = get_context(idx)
        prompt = document['text']
        tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
        logits = model(**tokens).logits
        targets = tokens.input_ids
        ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
        ls_l = ls[l-1]
        ls_l.backward()
        g = get_flattened_gradient(model, highsignal_names)
        # g = torch.cat([g, g.abs()])
        Gi[i] = g
    Gi = F.normalize(Gi, p=2, dim=1)
    # Gi = Gi - Gi.mean(dim=1, keepdim=True)
    j_index = blocks.index(iblock)
    jouter = sum(len(block) for block in blocks[:j_index])
    for jblock in tqdm(blocks[j_index:], leave=False):
        Gj = torch.zeros((len(jblock), len_g), device=device)
        for j, idx in enumerate(jblock):
            model.zero_grad()
            document, l = get_context(idx)
            prompt = document['text']
            tokens = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True).to(device)
            logits = model(**tokens).logits
            targets = tokens.input_ids
            ls = torch.nn.functional.cross_entropy(logits[0, :-1, :], targets[0, 1:], reduction='none')
            ls_l = ls[l-1]
            ls_l.backward()
            g = get_flattened_gradient(model, highsignal_names)
            # g = torch.cat([g, g.abs()])
            Gj[j] = g
        Gj = F.normalize(Gj, p=2, dim=1)
        # Gj = Gj - Gj.mean(dim=1, keepdim=True)
        Cij = torch.matmul(Gi, Gj.T)
        C[iouter:iouter+len(iblock), jouter:jouter+len(jblock)] = Cij
        C[jouter:jouter+len(jblock), iouter:iouter+len(iblock)] = Cij.T
        Cij_abs = torch.matmul(Gi.abs(), Gj.T.abs())
        C_abs[iouter:iouter+len(iblock), jouter:jouter+len(jblock)] = Cij_abs
        C_abs[jouter:jouter+len(jblock), iouter:iouter+len(iblock)] = Cij_abs.T
        jouter += len(jblock)
    iouter += len(iblock)

torch.save((idxs, C, C_abs), f"/om/user/ericjm/results/the-everything-machine/clustering-0/full_more_more.pt")
