#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np

def extract_feature(i):
  X = M[i+8:i+10, 3:12].flatten().astype("float")
  y_hat = float(M[i+9, 12])
  return X, y_hat

infile1, infile2, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

M = pd.read_csv(infile1, encoding='big5').as_matrix()

# ydata = b + w * xdata
b = 0.0
w = np.zeros((1, 18))
lr = 0.000000001
epoch = 700

# extract feature into x_data, y_data
i = 0
x_data = []
y_data = []
while i < M.shape[0]:
  X, y_hat = extract_feature(i)
  x_data.append(X)
  y_data.append(y_hat)
  i += 18

for e in range(epoch):
  b_grad = 0.0
  w_grad = np.zeros((1, 2*9))
  for n in range(len(x_data)):
    b_grad = b_grad  - 2*(y_data[n] - b - np.dot(w, x_data[n]))*1
    w_grad = w_grad  - 2*(y_data[n] - b - np.dot(w, x_data[n]))*x_data[n]

  # Print loss
  # print('epoch:{}\n Loss:{}'.format(e, (y_data[n] - b - np.dot(w, x_data[n]))**2))

  # Update parameters.
  b = b - lr * b_grad
  w = w - lr * w_grad


# Test
with open('output.csv', 'w') as f:
  f.write('id,value\n')
  M = pd.read_csv(infile2, header=None, encoding='big5').as_matrix()

  i = 0
  while i < M.shape[0]:
    X = M[i+8:i+10, 2:].flatten().astype("float")
    y = M[i, 0]
    f.write('{},{}\n'.format(y, (b + np.dot(w, X))[0]))
    i += 18