#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

def extract_feature(M, features, squares):
  x_data = []
  y_data = []
  for month in range(M.shape[0]):
    for i in range(M.shape[2]-10+1):
      X = M[month, features, i:i+9].flatten()
      Y = M[month, squares, i:i+9].flatten()
      Z = np.concatenate((X, Y**2), axis=0)
      x_data.append(Z)
      y_data.append(M[month, 9, i+9])
  return np.array(x_data), np.array(y_data)

# Start Program
infile1, infile2, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

# preprocessing on infile1
M = pd.read_csv(infile1, encoding='big5').as_matrix() #shape: (4320, 27)
M = M[:, 3:] #shape: (4320, 24)
M = np.reshape(M, (12, -1, 18, 24)) #shape: (12, 20, 18, 24)
M = M.swapaxes(1, 2).reshape(12, 18, -1) #shape: (12, 18, 480)
M[M == 'NR'] = '0.0'
M = M.astype(float)


# extract feature into x_data <shape:(5652, 9*len)>, y_data <shape:(5652,)>
feature_sieve = [7, 8, 9, 10, 14, 15, 16, 17]
square_sieve = [8, 9]
length = len(feature_sieve) + len(square_sieve)
x_data, y_data = extract_feature(M, feature_sieve, square_sieve)

# scaling
mean = np.mean(x_data, axis=0)
std = np.std(x_data, axis=0)
x_data = (x_data - mean) / (std + 1e-20)

# add a column for x_data at the front
one = np.ones((x_data.shape[0], 1))
x_data = np.concatenate((one, x_data), axis=1)

# Solve least square error by formula
ans = np.linalg.lstsq(x_data, y_data)
W = ans[0]

# Test

## check the folder of out.csv is exist; otherwise, make it
ensure_dir(outfile)

## save the parameter b, w
para = outfile.replace('csv', 'para')
np.savetxt(para, W.reshape((1, -1)), delimiter=',')

with open(outfile, 'w+') as f:
  M = pd.read_csv(infile2, header=None, encoding='big5').as_matrix()
  M = M[:, 2:] #shape: (4320, 9)
  M = M.reshape(-1, 18, 9) #shape: (240, 18, 9)
  M[M == 'NR'] = '0.0'
  M = M.astype(float)

  selected = feature_sieve
  square_selected = square_sieve

  f.write('id,value\n')
  for i in range(M.shape[0]):
    X = M[i, selected, :].flatten()
    Y = M[i, square_selected, :].flatten()
    Z = np.concatenate((X, Y**2), axis=0)
    Z = (Z - mean) / (std + 1e-20)

    # add 1 row for Z
    one = np.array([1.0])
    Z = np.concatenate((one, Z), axis=0)

    f.write('id_{},{}\n'.format(i, np.dot(W, Z)))
