
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
import torch

# from datasets import load_dataset
# from transformers import AutoTokenizer

curves = np.load("/om/user/ericjm/results/the-everything-machine/pythia-2/pythia-2.npy")

monogenic_idxs = {i: defaultdict(list) for i in [1, 2, 3, 4, 5]}

# let's have the intervals between plateau boundaries be 0.7 nats
upper_plateau_lowerbounds = np.arange(2.7, 11, 0.2).tolist()

for upper_plateau_lowerbound in tqdm(upper_plateau_lowerbounds):
    lower_plateau_lowerbounds = np.arange(0.0, upper_plateau_lowerbound-2.0, 0.2)
    for lower_plateau_lowerbound in tqdm(lower_plateau_lowerbounds, leave=False):
        aboves_upper_plateau_lowerbound = curves > upper_plateau_lowerbound
        belows_upper_plateau_upperbound = curves < (upper_plateau_lowerbound + 0.7)
        aboves_lower_plateau_lowerbound = curves > lower_plateau_lowerbound
        belows_lower_plateau_upperbound = curves < (lower_plateau_lowerbound + 0.7)
        key = ((upper_plateau_lowerbound, upper_plateau_lowerbound + 0.7), (lower_plateau_lowerbound, lower_plateau_lowerbound + 0.7))
        for j in tqdm([1, 2, 3, 4, 5], leave=False):
            within_before_j_above = np.all(aboves_upper_plateau_lowerbound[:, :j], axis=1) #.nonzero()
            within_before_j_below = np.all(belows_upper_plateau_upperbound[:, :j], axis=1) #.nonzero()
            # within_upper_plateau_range = np.intersect1d(within_before_j_above, within_before_j_below)
            within_after_j_above = np.all(aboves_lower_plateau_lowerbound[:, :j], axis=1) #.nonzero()
            within_after_j_below = np.all(belows_lower_plateau_upperbound[:, :j], axis=1) #.nonzero()
            # within_lower_plateau_range = np.intersect1d(within_after_j_above, within_after_j_below) 
            # monogenic_idxs[j][key].extend(np.intersect1d(within_upper_plateau_range, within_lower_plateau_range).tolist())
            intersection, = (within_before_j_above * within_before_j_below * within_after_j_above * within_after_j_below).nonzero()
            monogenic_idxs[j][key].extend(intersection.tolist())

torch.save(monogenic_idxs, "/om/user/ericjm/results/the-everything-machine/pythia-2/monogenic_idxs.pt")

