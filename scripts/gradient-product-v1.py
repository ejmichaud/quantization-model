
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import scipy.linalg
import torch
import torch.nn.functional as F
import sklearn.cluster

from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

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


dataset = dataset.map(tokenize_sample)
dataset = dataset.map(lambda sample: {"split_by_token": tokenizer.batch_decode(sample["input_ids"][0])})
dataset = dataset.map(lambda sample: {"tokens_len": len(sample["input_ids"][0])})
dataset = dataset.map(lambda sample: {"preds_len": max(sample["tokens_len"] - 1, 0)}) # fixed this on 2023-02-06 to accomodate empty documents
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


zero_idxs = curves[:, 0] < 0.1
zero_idxs, = zero_idxs.nonzero()
len(zero_idxs)
print_context(zero_idxs[0])
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
idxs = non_induction_zeros[:10000]
S = len(idxs)

block_len = 200
blocks = [idxs[i:min(len(idxs), i+block_len)] for i in range(0, len(idxs), block_len)]

C = torch.zeros((S, S), device=device)

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
        Gi[i] = g
    Gi = F.normalize(Gi, p=2, dim=1)
    jouter = 0
    for jblock in tqdm(blocks, leave=False):
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
            Gj[j] = g
        Gj = F.normalize(Gj, p=2, dim=1)
        Cij = torch.matmul(Gi, Gj.T)
        C[iouter:iouter+len(iblock), jouter:jouter+len(jblock)] = Cij
        jouter += len(jblock)
    iouter += len(iblock)

torch.save(C.detach().cpu(), "/om/user/ericjm/results/the-everything-machine/misc/C-v1.pt")


