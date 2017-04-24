#!/usr/bin/env python
import sys, os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs = 300
zoom_range = 0.2
model_name = 'dnn1.h5'
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
  Y = to_categorical(Y, num_classes)

# Change data into pictures format for generator
X = X.reshape(X.shape[0], height, width, 1)

# Split data
if isValid:
  valid_num = 3000
  X_train, Y_train = X[:-valid_num], Y[:-valid_num]
  X_valid, Y_valid = X[:-valid_num], Y[:-valid_num]
else:
  X_train, Y_train = X, Y

# Image PreProcessing
train_gen = ImageDataGenerator(rotation_range=25,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=[1-zoom_range, 1+zoom_range],
                              horizontal_flip=True)
train_gen.fit(X_train)


# Construct the model
model = Sequential()
model.add(Flatten(input_shape=input_shape))

model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint('dnn/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('dnn_log.csv', separator=',', append=False)
callbacks.append(csv_logger)

# Fit the model
if isValid:
  model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=10*X_train.shape[0]//batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(X_valid, Y_valid))
else:
  model.fit_generator(train_gen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=10*X_train.shape[0]//batch_size,
                    epochs=epochs,
                    callbacks=callbacks)

# Save model
model.save(model_name)
