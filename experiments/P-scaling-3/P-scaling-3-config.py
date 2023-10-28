
from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(5), np.log2(500), 20))]

if __name__ == '__main__':
    task_idx = int(sys.argv[1])
    width = widths[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v1.py \
                                -F /om2/user/ericjm/the-everything-machine/results/P-scaling-3 \
                                run with \
                                alpha=1.3\
                                batch_size=50000 \
                                width={width} \
                                depth=2 \
                                k=3 \
                                n=100 \
                                n_tasks=10000 \
                                steps=200000 \
                                log_freq=50 \
                                test_points=100000 \
                                seed=0
                                """)
