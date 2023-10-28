
from itertools import product
import os
import sys

import numpy as np

widths = [int(w) for w in np.power(2, np.linspace(np.log2(5), np.log2(1000), 15))]

if __name__ == '__main__':
    task_idx = int(sys.argv[1])
    width = widths[task_idx]
    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/the-everything-machine/scripts/sparse-parity-v1.py \
                                -F /om/user/ericjm/results/the-everything-machine/P-scaling-1 \
                                run with \
                                alpha=1.5 \
                                batch_size=10000 \
                                width={width} \
                                depth=2 \
                                k=2 \
                                n=50 \
                                n_tasks=300 \
                                steps=100000 \
                                log_freq=50 \
                                test_points=30000
                                seed=0
                                """)
