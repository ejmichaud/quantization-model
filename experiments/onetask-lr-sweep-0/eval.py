
import random
import time

from itertools import product
import os
import sys

import numpy as np

lrs = [2**x for x in range(-12, 4)] # 16

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(task_idx * 5)    

    lr  = lrs[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v4.py \
                                -F /om/user/ericjm/results/the-everything-machine/onetask-lr-sweep-0 \
                                run with \
                                alpha=1.4 \
                                D=-1 \
                                batch_size=10000 \
                                width=100 \
                                depth=2 \
                                k=3 \
                                n=50 \
                                n_tasks=1 \
                                steps=20000 \
                                lr={lr} \
                                log_freq=50 \
                                test_points=5000 \
                                seed=0
                                """)

