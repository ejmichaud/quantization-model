
import random
import time

from itertools import product
import os
import sys

import numpy as np

Ds = [int(D) for D in np.power(2, np.linspace(np.log2(10000), np.log2(5000000), 30))]

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(task_idx * 1)    

    D = Ds[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/D-scaling-3 \
                                run with \
                                alpha=1.5 \
                                offset=10 \
                                D={D} \
                                batch_size=100000 \
                                width=1000 \
                                depth=2 \
                                k=3 \
                                n=100 \
                                n_tasks=200 \
                                steps=100000 \
                                log_freq=1000 \
                                test_points=1000000 \
                                test_points_per_task=1000 \
                                seed=0
                                """)

