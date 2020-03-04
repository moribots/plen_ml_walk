#!/usr/bin/env python
import time
import numpy as np

# loop_time = 1.0 / 60.

# while True:
#     start_time = time.time()
#     time.sleep(1.0 / 120.0)
#     elapsed_time = time.time() - start_time
#     # print(elapsed_time)
#     if loop_time > elapsed_time:
#         print("EXTRA SLEEP")
#         time.sleep(loop_time - elapsed_time)
#     print("RATE: {}".format(1.0 / (time.time() - start_time)))

commands = np.array([])

# Number of data points to collect
num_iters = 100

for i in range(num_iters):
    commanded_value = (-np.pi / 2.0) + (i * (np.pi) / float(num_iters - 1))
    commands = np.append(commands, commanded_value)

print("CMDS: {}",format(commands))
