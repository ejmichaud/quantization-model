
import random
import time

from itertools import product
import os
import sys

import numpy as np

Ds = [int(D) for D in np.power(2, np.linspace(np.log2(10000), np.log2(5000000), 10))]
alphas = [1.2, 1.3, 1.4, 1.5]
configs = list(product(Ds, alphas))

if __name__ == '__main__':

    task_idx = int(sys.argv[1])

    D, alpha = configs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/D-scaling-7 \
                                run with \
                                alpha={alpha} \
                                D={D} \
                                batch_size=20000 \
                                width=500 \
                                depth=2 \
                                k=3 \
                                n=100 \
                                n_tasks=500 \
                                steps=500000 \
                                log_freq=500 \
                                test_points=1000000 \
                                seed=0 \
								""")

