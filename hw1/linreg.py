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
  for month in range(M.shape[0]):
    for i in range(M.shape[2]-10+1):
      x_data.append(M[month, features, i:i+9].flatten().astype("float"))
      y_data.append(float(M[month, 9, i+9]))
  return np.array(x_data), np.array(y_data)

# Start Program
infile1, infile2, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

# preprocessing on infile1
M = pd.read_csv(infile1, encoding='big5').as_matrix() #shape: (4320, 27)
M = M[:, 3:] #shape: (4320, 24)
M = np.reshape(M, (12, -1, 18, 24)) #shape: (12, 20, 18, 24)
M = M.swapaxes(1, 2).reshape(12, 18, -1) #shape: (12, 18, 480)
## map 'NR' -> '-1'
for month in range(12):
  M[month, 10, :] = np.array(list(map(lambda i: i if i != 'NR' else '-1', M[month, 10, :])))


# extract feature into x_data <shape:(5652, 9*18)>, y_data <shape:(5652,)>
feature_sieve = [i for i in range(0, 18)]
x_data, y_data = extract_feature(M, feature_sieve)

# ydata = b + w * xdata
b = 0.0
w = np.zeros(len(feature_sieve)*9)
lr = 3e-10
epoch = 100000

prev_res = 1e10
for e in range(epoch):
  b_grad = 0.0
  w_grad = np.zeros(len(feature_sieve)*9)
  res = 0.0

  error = y_data - b - np.dot(x_data, w) #shape: (5652,)

  b_grad = b_grad - 2*np.sum(error)*1 #shape: ()
  w_grad = w_grad - 2*np.dot(error, x_data) #shape: (162,)
  res = np.mean(np.square(error))

  # Print loss
  if e % 100 == 0:
    print('epoch:{}\n Loss:{}'.format(e, res))

  if prev_res - res < 1e-8: break
  prev_res = res

  # Update parameters.
  b = b - lr * b_grad
  w = w - lr * w_grad


# Test

## check the folder of out.csv is exist; otherwise, make it
ensure_dir(outfile)

## save the parameter b, w
para = outfile.replace('csv', 'para')
with open(para, 'w+') as f:
  f.write('{}\n'.format(b))
  f.write('{}\n'.format(','.join(list(map(lambda x: str(x), w.flatten())))))

with open(outfile, 'w+') as f:
  f.write('id,value\n')
  M = pd.read_csv(infile2, header=None, encoding='big5').as_matrix()

  i = 0
  while i < M.shape[0]:
    ## map 'NR' -> '-1'
    M[i+10, :] = np.array(list(map(lambda x: x if x != 'NR' else '-1', M[i+10, :])))

    modified_sieve = [n+i for n in feature_sieve]
    X = M[modified_sieve, 2:].flatten().astype("float")
    y = M[i, 0]
    f.write('{},{}\n'.format(y, b + np.dot(w, X)))
    i += 18