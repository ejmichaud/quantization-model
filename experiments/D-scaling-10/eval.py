
import random
import time

from itertools import product
import os
import sys

import numpy as np

Ds = [int(D) for D in np.power(2, np.linspace(np.log2(1000), np.log2(10000000), 25))]
alphas = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]
ks = [2, 3, 4, 5]
configs = list(product(ks, alphas, Ds)) # 1500

if __name__ == '__main__':

    task_idx = int(sys.argv[1])

    k, alpha, D = configs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v5.py \
                                -F /om/user/ericjm/results/the-everything-machine/D-scaling-10 \
                                run with \
                                alpha={alpha} \
                                D={D} \
                                batch_size=20000 \
                                width=1000 \
                                depth=2 \
                                k={k} \
                                n=100 \
                                n_tasks=300 \
                                steps=500000 \
                                stop_early=True \
                                log_freq=500 \
                                test_points=1000000 \
                                seed=0 \
								""")
