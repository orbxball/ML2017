#!/usr/bin/env python3
import os
import sys
import numpy as np

M1 = np.loadtxt(sys.argv[1], delimiter=',', ndmin=2)
M2 = np.loadtxt(sys.argv[2], delimiter=',', ndmin=2)

ans = list(np.matmul(M1, M2).flatten())
with open('ans_one.txt', 'w') as f:
  for i in sorted(ans):
    f.write(str(int(i))+'\n')
