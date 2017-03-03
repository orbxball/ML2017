#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def extract_feature(M, features):
  x_data = []
  y_data = []
  for i in range(M.shape[1]-10+1):
    x_data.append(M[features, i:i+9].flatten().astype("float"))
    y_data.append(float(M[9, i+9]))
  return x_data, y_data

# Start Program
infile1, infile2, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

# preprocessing on infile1
M = pd.read_csv(infile1, encoding='big5').as_matrix() #shape: (4320, 27)
M = M[:, 3:] #shape: (4320, 24)
M = np.reshape(M, (-1, 18, 24)) #shape: (240, 18, 24)
M = M.swapaxes(0, 1).reshape(18, -1) #shape: (18, 5670)

# extract feature into x_data, y_data
feature_sieve = [i for i in range(0, 18) if i != 10]
x_data, y_data = extract_feature(M, feature_sieve)

# ydata = b + w * xdata
b = 0.0
w = np.zeros((1, len(feature_sieve)*9))
lr = 2e-10
epoch = 10000

prev_res = 1e10
for e in range(epoch):
  b_grad = 0.0
  w_grad = np.zeros((1, len(feature_sieve)*9))
  res = 0.0
  for n in range(len(x_data)):
    b_grad = b_grad  - 2*(y_data[n] - b - np.dot(w, x_data[n]))*1
    w_grad = w_grad  - 2*(y_data[n] - b - np.dot(w, x_data[n]))*x_data[n]
    res += (y_data[n] - b - np.dot(w, x_data[n]))**2

  # Print loss
  print('epoch:{}\n Loss:{}'.format(e, res/len(x_data)))

  tmp = prev_res - res/len(x_data)
  if tmp[0] < 1e-8: break

  prev_res = res/len(x_data)

  # Update parameters.
  b = b - lr * b_grad
  w = w - lr * w_grad


# Test

## check the folder of out.csv is exist; otherwise, make it
ensure_dir(outfile)

with open(outfile, 'w+') as f:
  f.write('id,value\n')
  M = pd.read_csv(infile2, header=None, encoding='big5').as_matrix()

  i = 0
  while i < M.shape[0]:
    modified_sieve = [n+i for n in feature_sieve]
    X = M[modified_sieve, 2:].flatten().astype("float")
    y = M[i, 0]
    f.write('{},{}\n'.format(y, (b + np.dot(w, X))[0]))
    i += 18