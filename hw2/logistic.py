#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)

def sigmoid(z):
  return 1 / (1 + np.exp(-1 * z))

# Start Program
X_train, Y_train = sys.argv[1], sys.argv[2]
X_test, outfile = sys.argv[3], sys.argv[4]

X_train = pd.read_csv(X_train).as_matrix() #shape: (32561, 106)
Y_train = pd.read_csv(Y_train, header=None).as_matrix() #shape: (32561, 1)
X_test = pd.read_csv(X_test).as_matrix() #shape: (16281, 106)
Y_train = Y_train.reshape(Y_train.shape[0]) #shape: (32561,)

# scaling: only on features, not label
mean = np.mean(X_train, axis=0) #shape: (106,)
std = np.std(X_train, axis=0) #shape: (106,)
X_train = (X_train - mean) / (std + 1e-100)

# initialize
b = 0.0
w = np.ones(X_train.shape[1])
lr = 5e-1
epoch = 2800
b_lr = 0.0
w_lr = np.zeros(X_train.shape[1])

for e in range(epoch):
  # Calculate the value of error for loss function
  z = np.dot(X_train, w) + b #shape: (32561,)
  f = sigmoid(z) #shape: (32561,)
  error = Y_train - f #shape: (32561,)

  # Calculate gradient
  b_grad = -np.sum(error)*1 #shape: ()
  w_grad = -np.dot(error.T, X_train) #shape: (X_train.shape[1],)
  b_lr = b_lr + b_grad**2
  w_lr = w_lr + w_grad**2

  # calculate loss = cross entropy
  loss = -np.mean(Y_train*np.log(f+1e-100) + (1-Y_train)*np.log(1-f+1e-100))

  # Update parameters.
  b = b - lr/np.sqrt(b_lr) * b_grad
  w = w - lr/np.sqrt(w_lr) * w_grad

  # Print loss
  if (e+1) % 100 == 0:
    f[f >= 0.5] = 1
    f[f < 0.5] = 0
    acc = Y_train - f #shape: (32561,)
    acc[acc == 0] = 2
    acc[acc != 2] = 0
    print('epoch:{}\n Loss:{}\n Accuracy:{}%\n'.format(e+1, loss, np.sum(acc) * 50 / acc.shape[0]))


# Test

## check the folder of out.csv is exist; otherwise, make it
ensure_dir(outfile)

## save the parameter b, w
para = outfile.replace('csv', 'para')
with open(para, 'w+') as f:
  W = np.concatenate((b.reshape(-1), w), axis=0)
  np.savetxt(para, W.reshape((1, -1)), delimiter=',')

with open(outfile, 'w+') as file:
  file.write('id,label\n')
  ans = []
  for i in range(X_test.shape[0]):
    Z = X_test[i]
    Z = (Z - mean) / (std + 1e-100)
    z = np.dot(w, Z) + b
    if (z >= 0):
      ans.append('{},{}'.format(i+1, 1))
    else:
      ans.append('{},{}'.format(i+1, 0))
  file.write('\n'.join(ans))
