
import random
import time

from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(10), np.log2(800), 12))]
ks = [2, 3, 4, 5]
configs = list(product(widths, ks))

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(task_idx * 1)    

    width, k = configs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/P-scaling-10 \
                                run with \
                                alpha=1.2 \
                                D=-1 \
                                batch_size=50000 \
                                width={width} \
                                depth=2 \
                                k={k} \
                                n=50 \
                                n_tasks=400 \
                                steps=100000 \
                                log_freq=500 \
                                test_points=1000000 \
                                test_points_per_task=1000 \
                                seed=0
                                """)

