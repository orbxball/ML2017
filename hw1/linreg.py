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
      Z = np.concatenate((X, Y), axis=0)
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


# extract feature into x_data <shape:(5652, 9*18)>, y_data <shape:(5652,)>
feature_sieve = [8, 9, 10, 15, 16]
square_sieve = [8, 9, 10, 15, 16]
length = len(feature_sieve) + len(square_sieve)
x_data, y_data = extract_feature(M, feature_sieve, square_sieve)

# ydata = b + w * xdata
b = 0.0
w = np.zeros(length*9)
lr = 1e2
epoch = 200000
b_lr = 0.0
w_lr = np.zeros(length*9)

prev_loss = 1e10
for e in range(epoch):
  b_grad = 0.0
  w_grad = np.zeros(length*9)
  loss = 0.0

  # Calculate the value of the loss function
  error = y_data - b - np.dot(x_data, w) #shape: (5652,)

  # Calculate gradient
  b_grad = b_grad - 2*np.sum(error)*1 #shape: ()
  w_grad = w_grad - 2*np.dot(error, x_data) #shape: (162,)
  b_lr = b_lr + b_grad**2
  w_lr = w_lr + w_grad**2
  loss = np.mean(np.square(error))

  # Update parameters.
  b = b - lr/np.sqrt(b_lr) * b_grad
  w = w - lr/np.sqrt(w_lr) * w_grad

  # Print loss
  if (e+1) % 1000 == 0:
    print('epoch:{}\n Loss:{}'.format(e+1, np.sqrt(loss)))
  # if prev_loss - loss < 1e-8: break
  # prev_loss = loss


# Test

## check the folder of out.csv is exist; otherwise, make it
ensure_dir(outfile)

## save the parameter b, w
para = outfile.replace('csv', 'para')
with open(para, 'w+') as f:
  f.write('{}\n'.format(b))
  f.write('{}\n'.format(','.join(list(map(lambda x: str(x), w.flatten())))))

with open(outfile, 'w+') as f:
  M = pd.read_csv(infile2, header=None, encoding='big5').as_matrix()
  M = M[:, 2:] #shape: (4320, 9)
  M = M.reshape(-1, 18, 9) #shape: (240, 18, 9)
  M[M == 'NR'] = '0.0'
  M = M.astype(float)

  selected = [i for i in feature_sieve]
  square_selected = [i for i in square_sieve]

  f.write('id,value\n')
  for i in range(M.shape[0]):
    X = M[i, selected, :].flatten()
    Y = M[i, square_selected, :].flatten()
    Z = np.concatenate((X, Y), axis=0)
    f.write('id_{},{}\n'.format(i, b + np.dot(w, Z)))