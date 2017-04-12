#!/usr/bin/env python
import sys, os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import losses
from keras import optimizers

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 32
epochs = 20
pool_size = (2, 2)
model_name = 'cnn_d3.h5'
isValid = 1

# Read the train data
with open(sys.argv[1], "r+") as f:
  line = f.read().strip().replace(',', ' ').split('\n')[1:]
  raw_data = ' '.join(line)
  length = width*height+1 #1 is for label
  data = np.array(raw_data.split()).astype('float').reshape(-1, length)
  X = data[:, 1:]
  Y = data[:, 0]
  X /= 255
  Y = Y.reshape(Y.shape[0], 1)
  Y = keras.utils.to_categorical(Y, num_classes)

# Change data into CNN format
X = X.reshape(X.shape[0], height, width, 1)

# Split data
if isValid:
  valid_num = 1000
  X_train, Y_train = X[:-valid_num], Y[:-valid_num]
  X_valid, Y_valid = X[-valid_num:], Y[-valid_num:]
else:
  X_train, Y_train = X, Y

# Construct the model
model = Sequential()
model.add(Conv2D(25, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Conv2D(50, (3, 3), activation='relu'))
model.add(Conv2D(100, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=pool_size))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the model
if isValid:
  model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_valid, Y_valid))
else:
  model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

# Evaluate the test data
if isValid:
  score = model.evaluate(X_valid, Y_valid, verbose=0)
else:
  score = model.evaluate(X_train, Y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save model
model.save(model_name)
