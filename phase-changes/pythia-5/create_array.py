import os
import re
import numpy as np
from tqdm.auto import tqdm
import torch

steps = list(range(1000, 144000, 1000))
results_dir = '/om/user/ericjm/results/phase-changes/pythia-5/'

checkpoint_files = [f for f in os.listdir(results_dir) if re.fullmatch(r"step\d+.pt", f)]
print(len(checkpoint_files))
assert len(checkpoint_files) == 143

checkpoint_files = [f"step{step}.pt" for step in steps]
# checkpoint_files = [f for f in os.listdir('/om/user/ericjm/results/phase-changes/pythia-5') if f != 'logs']
# checkpoint_files = sorted(checkpoint_files, key=lambda c: int(c.split('.pt')[0]))
# print(checkpoint_files[:20])
# example_data = torch.load("/om/user/ericjm/results/phase-changes/pythia-5/step1000.pt")
example_data = torch.load(os.path.join(results_dir, checkpoint_files[0]))

T = len(checkpoint_files) # number of checkpoints/timesteps (143)
D = sum([len(x) for x in example_data]) # number of tokens (~1.6M)

timeseries = np.zeros((D, T), dtype=np.float32)

for i, ckpt_file in tqdm(list(enumerate(checkpoint_files))):
    ckpt_path = os.path.join(results_dir, ckpt_file)
    try:
        ckpt_results = torch.load(ckpt_path)
        j = 0        
        for x in ckpt_results:
            timeseries[j:j+len(x), i] = np.array(x, dtype=np.float32)
            j += len(x)        
    except Exception as e:
        print(e)
        print(i, ckpt_file)

np.save(os.path.join(results_dir, 'pythia-5.npy'), timeseries)

