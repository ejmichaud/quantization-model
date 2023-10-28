
import random
import time

from itertools import product
import os
import sys

import numpy as np

Ds = [int(D) for D in np.power(2, np.linspace(np.log2(10000), np.log2(5000000), 25))]
alphas = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]
weight_decays = np.power(2, np.linspace(np.log2(0.001), np.log2(10), 30))

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    D = Ds[-3]
    alpha = 1.2
    wd = weight_decays[task_idx]
    
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/weight-decay-0 \
                                run with \
                                alpha={alpha} \
                                D={D} \
                                batch_size=20000 \
                                width=500 \
                                depth=2 \
                                k=3 \
                                n=100 \
                                n_tasks=500 \
                                weight_decay={wd} \
                                steps=100000 \
                                log_freq=500 \
                                test_points=1000000 \
                                seed=0 \
								""")

