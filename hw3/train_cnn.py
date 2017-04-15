#!/usr/bin/env python
import sys, os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import losses
from keras import optimizers

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 100
zoom_range = 0.05
model_name = 'pre2.h5'
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

# Split the data
if isValid:
  valid_num = 3000
  permu = np.random.permutation(X.shape[0])
  X_train, Y_train = X[permu[:-valid_num]], Y[permu[:-valid_num]]
  X_valid, Y_valid = X[permu[-valid_num:]], Y[permu[-valid_num:]]
else:
  X_train, Y_train = X, Y

# Construct the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(LeakyReLU(alpha=0.03))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.03))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.03))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3)))
model.add(LeakyReLU(alpha=0.03))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Image PreProcessing
train_gen = ImageDataGenerator(rotation_range=10,
                              width_shift_range=0.05,
                              height_shift_range=0.05,
                              shear_range=0.05,
                              zoom_range=[1-zoom_range, 1+zoom_range],
                              horizontal_flip=True)
train_gen.fit(X_train)

# Fit the model
if isValid:
  model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=10*X_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=(X_valid, Y_valid))
else:
  model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=10*X_train.shape[0]//batch_size,
                    epochs=epochs)

# Save model
model.save(model_name)
