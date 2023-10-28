
import random
import time

from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(10), np.log2(500), 20))]

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(task_idx * 1)    

    width = widths[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v3.py \
                                -F /om/user/ericjm/results/the-everything-machine/P-scaling-8 \
                                run with \
                                alpha=1.1 \
                                D=-1 \
                                batch_size=30000 \
                                width={width} \
                                depth=2 \
                                k=2 \
                                n=50 \
                                n_tasks=800 \
                                steps=100000 \
                                log_freq=100 \
                                test_points=5000 \
                                test_points_per_task=1000 \
                                seed=0
                                """)

