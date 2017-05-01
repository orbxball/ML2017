#!/usr/bin/env python
import sys, os
import numpy as np
from keras.models import load_model

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
model_name1 = 'pre6.h5'
model_name2 = 'weights.151-0.71967.h5'
model_name3 = 'ensemble1.h5'
model_name4 = 'weights.031-0.71633.h5'
model_name5 = 'more-filters.h5'

# Read the test data
with open(sys.argv[1], "r+") as f:
  line = f.read().strip().replace(',', ' ').split('\n')[1:]
  raw_data = ' '.join(line)
  length = width*height+1 #1 is for label
  data = np.array(raw_data.split()).astype('float').reshape(-1, length)
  X = data[:, 1:]
  X /= 255

# Load model
model_path = 'model/'
model_1 = load_model(os.path.join(model_path, model_name1))
model_2 = load_model(os.path.join(model_path, model_name2))
model_3 = load_model(os.path.join(model_path, model_name3))
model_4 = load_model(os.path.join(model_path, model_name4))
model_5 = load_model(os.path.join(model_path, model_name5))

# Plot model
# plot_model(model,to_file='cnn_model.png')

# Predict the test data
X = X.reshape(X.shape[0], height, width, 1)
print('Predicting model 1: {}'.format(model_name1))
ans_1 = model_1.predict(X)
print('Predicting model 2: {}'.format(model_name2))
ans_2 = model_2.predict(X)
print('Predicting model 3: {}'.format(model_name3))
ans_3 = model_3.predict(X)
print('Predicting model 4: {}'.format(model_name4))
ans_4 = model_4.predict(X)
print('Predicting model 5: {}'.format(model_name5))
ans_5 = model_5.predict(X)
ans = ans_1 + ans_2 + ans_3 + ans_4 + ans_5
ans = np.argmax(ans, axis=-1)
ans = list(ans)

# Write prediction
## check the folder of out.csv is exist; otherwise, make it
ensure_dir(sys.argv[2])

result = []
for index, value in enumerate(ans):
  result.append("{0},{1}".format(index, value))

with open(sys.argv[2], "w+") as f:
  f.write("id,label\n")
  f.write("\n".join(result))
