import os
import re
import numpy as np
from tqdm.auto import tqdm
import torch

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

results_dir = '/om/user/ericjm/results/the-everything-machine/pythia-1/'

checkpoint_files = [f for f in os.listdir(results_dir) if re.fullmatch(r".*\.pt", f)]
print(len(checkpoint_files))
print(checkpoint_files)
assert len(checkpoint_files) == 7
checkpoint_files = [f"{model_name}.pt" for model_name in model_names[:-1]] # so that the order is right

# checkpoint_files = [f for f in os.listdir('/om/user/ericjm/results/phase-changes/pythia-2') if f != 'logs']
# checkpoint_files = sorted(checkpoint_files, key=lambda c: int(c.split('.pt')[0]))
# print(checkpoint_files[:20])
# example_data = torch.load("/om/user/ericjm/results/phase-changes/pythia-2/step1000.pt")

example_data = torch.load(os.path.join(results_dir, checkpoint_files[0]))
T = len(checkpoint_files) # number of checkpoints (7)
D = sum([len(x) for x in example_data]) # number of tokens (~10M)
timeseries = np.zeros((D, T))

for i, ckpt_file in tqdm(list(enumerate(checkpoint_files))):
    ckpt_path = os.path.join(results_dir, ckpt_file)
    try:
        ckpt_results = torch.load(ckpt_path)
        j = 0
        for x in ckpt_results:
            timeseries[j:j+len(x), i] = np.array(x)
            j += len(x)
    except Exception as e:
        print(e)
        print(i, ckpt_file)

np.save(os.path.join(results_dir, 'pythia-1.npy'), timeseries)

