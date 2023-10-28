
import random
import time

from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(10), np.log2(500), 20))]
alphas = [1.4, 1.8, 2.2]
seeds = [0, 1, 2]
configs = list(product(widths, alphas, seeds))

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    time.sleep(task_idx * 5)    

    width, alpha, seed = configs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v1.py \
                                -F /om/user/ericjm/results/the-everything-machine/P-scaling-6 \
                                run with \
                                alpha={alpha} \
                                batch_size=20000 \
                                width={width} \
                                depth=2 \
                                k=3 \
                                n=100 \
                                n_tasks=200 \
                                steps=200000 \
                                log_freq=50 \
                                test_points=60000 \
                                seed={seed}
                                """)

