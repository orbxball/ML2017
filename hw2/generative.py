#!/usr/bin/env python
import sys, os
import numpy as np
import pandas as pd

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)

def halfGaussianDistribution(mean, cov, x):
  t = -1 / 2 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
  return 1 / (((np.linalg.det(cov)))**0.5) * np.exp(t)

# Start Program
X_train, Y_train = sys.argv[1], sys.argv[2]
X_test, outfile = sys.argv[3], sys.argv[4]

X_train = pd.read_csv(X_train).as_matrix() #shape: (32561, 106)
Y_train = pd.read_csv(Y_train, header=None).as_matrix() #shape: (32561, 1)
X_test = pd.read_csv(X_test).as_matrix() #shape: (16281, 106)
Y_train = Y_train.reshape(Y_train.shape[0]) #shape: (32561,)

# Seperate class A(1) & class B(0)
Apicker = (Y_train == 1)
Bpicker = (Y_train == 0)
A_train = X_train[Apicker, :]
B_train = X_train[Bpicker, :]

# P(C1) = probA; P(C2) = probB
probA = np.sum(Y_train) / Y_train.shape[0]
probB = 1 - probA

# C1: mean & covariance
meanA = np.mean(A_train, axis=0)
covA = np.dot((A_train - meanA).T, A_train - meanA) / A_train.shape[0]

# C2: mean & covariance
meanB = np.mean(B_train, axis=0)
covB = np.dot((B_train - meanB).T, B_train - meanB) / B_train.shape[0]

cov = probA * covA + probB * covB

# loss
acc = 0
for i in range(X_train.shape[0]):
  x = X_train[i]
  # P(x|C1)
  PconA = halfGaussianDistribution(meanA, cov, x)
  # P(x|C2)
  PconB = halfGaussianDistribution(meanB, cov, x)
  nominator = np.log(probA * PconA + 1e-100)
  denominator = np.log(probA * PconA + probB * PconB + 1e-100)
  p = nominator - denominator

  if p >= np.log(0.5):
    if Y_train[i] == 1: acc += 1
  else:
    if Y_train[i] == 0: acc += 1
print('loss: {}'.format(acc / X_train.shape[0]))

with open(outfile, 'w+') as file:
  file.write('id,label\n')
  ans = []
  for i in range(X_test.shape[0]):
    x = X_test[i]
    # P(x|C1)
    PconA = halfGaussianDistribution(meanA, cov, x)
    # P(x|C2)
    PconB = halfGaussianDistribution(meanB, cov, x)
    nominator = np.log(probA * PconA + 1e-100)
    denominator = np.log(probA * PconA + probB * PconB + 1e-100)
    p = nominator - denominator

    if p >= np.log(0.5):
      ans.append('{},{}'.format(i+1, 1))
    else:
      ans.append('{},{}'.format(i+1, 0))
  file.write('\n'.join(ans))