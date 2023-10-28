#!/usr/bin/env python
# coding: utf-8
"""
This script trains MLPs on multiple sparse parity problems at once.

Comments
    - Ways that it doesn't actually do sampling:
        (1) In the infinite data regime, frequencies within a batch are given exactly by the power law, not via sampling
        (2) In the finite data regime, the frequencies within the train set are given exactly by a power law, not via sampling
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
# import scipy.stats
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("sparse-parity-v3")
ex.captured_out_filter = apply_backspaces_and_linefeeds


def get_batch(n_tasks, n, Ss, codes, sizes, device='cpu', dtype=torch.float64):
    """Creates batch. 

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    n : int
        Bit string length for sparse parity problem.
    Ss : list of lists of ints
        Subsets of [1, ... n] to compute sparse parities on.
    codes : list of int
        The subtask indices which the batch will consist of
    sizes : list of int
        Number of samples for each subtask
    device : str
        Device to put batch on.
    dtype : torch.dtype
        Data type to use for input x. Output y is torch.int64.

    Returns
    -------
    x : torch.Tensor
        inputs
    y : torch.Tensor
        labels
    """
    batch_x = torch.zeros((sum(sizes), n_tasks+n), device=device)
    batch_y = torch.zeros((sum(sizes),), dtype=torch.int64, device=device)
    start_i = 0
    for (S, size, code) in zip(Ss, sizes, codes):
        if size > 0:
            x = torch.randint(low=0, high=2, size=(size, n), device=device)
            y = torch.sum(x[:, S], dim=1) % 2
            x_task_code = torch.zeros((size, n_tasks), device=device)
            x_task_code[:, code] = 1
            x = torch.cat([x_task_code, x], dim=1)
            x = x.to(torch.float64)
            batch_x[start_i:start_i+size, :] = x
            batch_y[start_i:start_i+size] = y
            start_i += size
    return batch_x.to(dtype), batch_y
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    n_tasks = 100
    n = 50
    k = 3
    alpha = 1.5

    D = -1 # -1 for infinite data

    width = 100
    depth = 2
    activation = 'ReLU'
    
    steps = 25000
    batch_size = 10000
    lr = 1e-3
    test_points = 30000
    test_points_per_task = 1000
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    log_freq = max(1, steps // 1000)
    verbose=False

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(n_tasks,
        n,
        k,
        alpha,
        D,
        width,
        depth,
        activation,
        test_points,
        test_points_per_task,
        steps,
        batch_size,
        lr,
        device,
        dtype,
        log_freq,
        verbose,
        seed,
        _log):

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if activation == 'ReLU':
        activation_fn = nn.ReLU
    elif activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"

    # create model
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(n_tasks + n, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 2))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)
    _log.debug("Created model.")
    _log.debug(f"Model has {sum(t.numel() for t in mlp.parameters())} parameters") 
    ex.info['P'] = sum(t.numel() for t in mlp.parameters())

    Ss = []
    for _ in range(n_tasks * 10):
        S = tuple(sorted(list(random.sample(range(n), k))))
        if S not in Ss:
            Ss.append(S)
        if len(Ss) == n_tasks:
            break
    assert len(Ss) == n_tasks, "Couldn't find enough subsets for tasks for the given n, k"
    ex.info['Ss'] = Ss

    Z = np.sum(np.power(np.arange(1, n_tasks+1, 1, dtype=np.float64), -alpha))
    probs = np.power(np.arange(1, n_tasks+1, 1, dtype=np.float64), -alpha) / Z

    batch_sizes = [int(prob * batch_size) for prob in probs]
    test_batch_sizes = [int(prob * test_points) for prob in probs]
    _log.debug(f"Total batch size = {sum(batch_sizes)}")

    if D != -1:
        train_set_sizes = [int(prob * D) for prob in probs]
        train_x, train_y = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=train_set_sizes, device='cpu', dtype=dtype)
        train_data = torch.utils.data.TensorDataset(train_x, train_y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=min(D, batch_size), shuffle=True)
        train_iter = cycle(train_loader)
        ex.info['D'] = len(train_data)
    else:
        ex.info['D'] = steps * batch_size

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    ex.info['log_steps'] = list()
    ex.info['accuracies'] = list()
    ex.info['losses'] = list()
    ex.info['losses_subtasks'] = dict()
    ex.info['accuracies_subtasks'] = dict()
    for i in range(n_tasks):
        ex.info['losses_subtasks'][str(i)] = list()
        ex.info['accuracies_subtasks'][str(i)] = list()
    for step in tqdm(range(steps), disable=not verbose):
        if step % log_freq == 0:
            with torch.no_grad():
                x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=test_batch_sizes, device=device, dtype=dtype)
                y_i_pred = mlp(x_i)
                labels_i_pred = torch.argmax(y_i_pred, dim=1)
                ex.info['accuracies'].append(torch.sum(labels_i_pred == y_i).item() / test_points) 
                ex.info['losses'].append(loss_fn(y_i_pred, y_i).item())
                for i in range(n_tasks):
                    x_i, y_i = get_batch(n_tasks=n_tasks, n=n, Ss=[Ss[i]], codes=[i], sizes=[test_points_per_task], device=device, dtype=dtype)
                    y_i_pred = mlp(x_i)
                    ex.info['losses_subtasks'][str(i)].append(loss_fn(y_i_pred, y_i).item())
                    labels_i_pred = torch.argmax(y_i_pred, dim=1)
                    ex.info['accuracies_subtasks'][str(i)].append(torch.sum(labels_i_pred == y_i).item() / test_points_per_task)
                ex.info['log_steps'].append(step)
        optimizer.zero_grad()
        if D == -1:
            x, y_target = get_batch(n_tasks=n_tasks, n=n, Ss=Ss, codes=list(range(n_tasks)), sizes=batch_sizes, device=device, dtype=dtype)
        else:
            x, y_target = next(train_iter)
            x = x.to(device)
            y_target = y_target.to(device)
        y_pred = mlp(x)
        loss = loss_fn(y_pred, y_target)
        loss.backward()
        optimizer.step()

