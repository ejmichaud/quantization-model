
import random
import time

from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(10), np.log2(1000), 20))]

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(task_idx * 1)    

    width = widths[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/P-scaling-11 \
                                run with \
                                alpha=1.3 \
                                D=-1 \
                                batch_size=100000 \
                                width={width} \
                                depth=2 \
                                k=4 \
                                n=50 \
                                n_tasks=500 \
                                steps=100000 \
                                log_freq=1000 \
                                test_points=1000000 \
                                test_points_per_task=1000 \
                                seed=0
                                """)

