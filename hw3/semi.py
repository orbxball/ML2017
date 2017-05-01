#!/usr/bin/env python
import sys, os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model

# Parameter
height = width = 48
num_classes = 7
input_shape = (height, width, 1)
batch_size = 128
epochs_1 = 30
epochs_2 = 50
zoom_range = 0.2
model_name = 'semi.h5'
isValid = 1

def construct_model(model):
  model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.2))

  model.add(Conv2D(128, (3, 3), padding='same'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.25))

  model.add(Conv2D(256, (3, 3), padding='same'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.3))

  model.add(Conv2D(512, (3, 3), padding='same'))
  model.add(LeakyReLU(alpha=0.03))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
  model.add(Dropout(0.4))

  model.add(Flatten())

  model.add(Dense(256, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  model.summary()

def compile_model(model):
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


# Read the train data
try:
  X = np.load('X.npy')
  Y = np.load('Y.npy')
except:
  with open(sys.argv[1], "r+") as f:
    line = f.read().strip().replace(',', ' ').split('\n')[1:]
    raw_data = ' '.join(line)
    length = width*height+1 #1 is for label
    data = np.array(raw_data.split()).astype('float').reshape(-1, length)
    X = data[:, 1:]
    Y = data[:, 0]
  # Change data into CNN format
  X = X.reshape(X.shape[0], height, width, 1)
  Y = Y.reshape(Y.shape[0], 1)
  print('Saving X.npy & Y.npy')
  np.save('X.npy', X) # (-1, height, width, 1)
  np.save('Y.npy', Y) # (-1, 1)

X /= 255
Y = to_categorical(Y, num_classes)


# Split the data
if isValid:
  valid_num = 3000
  X_train, Y_train = X[:-valid_num], Y[:-valid_num]
  X_valid, Y_valid = X[-valid_num:], Y[-valid_num:]

else:
  X_train, Y_train = X, Y

print(X_train.shape)
print(Y_train.shape)
print(X_valid.shape)
print(Y_valid.shape)

## split the label && unlabel
split_point = 20000
X_label, Y_label = X_train[:-split_point], Y_train[:-split_point]
X_unlabel = X_train[-split_point:]
print(X_label.shape)
print(Y_label.shape)
print(X_unlabel.shape)


# Construct the model
model = Sequential()
construct_model(model)
compile_model(model)

# Image PreProcessing
train_gen = ImageDataGenerator(rotation_range=25,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.1,
                              zoom_range=[1-zoom_range, 1+zoom_range],
                              horizontal_flip=True)
train_gen.fit(X_label)

# Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint('semi1_ckpt/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('_semi_log.csv', separator=',', append=False)
callbacks.append(csv_logger)

# Fit the model
if isValid:
  model.fit_generator(train_gen.flow(X_label, Y_label, batch_size=batch_size),
                    steps_per_epoch=10*X_label.shape[0]//batch_size,
                    epochs=epochs_1,
                    callbacks=callbacks,
                    validation_data=(X_valid, Y_valid))
else:
  model.fit_generator(train_gen.flow(X_label, Y_label, batch_size=batch_size),
                    steps_per_epoch=10*X_label.shape[0]//batch_size,
                    epochs=epochs_1,
                    callbacks=callbacks)

# Save model
model.save('_{}'.format(model_name))
# model = load_model(model_name)

# Predict on unlabel data
Y_unlabel = model.predict_classes(X_unlabel)
Y_unlabel = to_categorical(Y_unlabel, num_classes)

new_X = np.concatenate((X_label, X_unlabel), axis=0)
new_Y = np.concatenate((Y_label, Y_unlabel), axis=0)

# Training 2
model2 = Sequential()
construct_model(model2)
compile_model(model2)

train_gen.fit(new_X)

# Callbacks
callbacks = []
modelcheckpoint = ModelCheckpoint('semi2_ckpt/weights.{epoch:03d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True)
callbacks.append(modelcheckpoint)
csv_logger = CSVLogger('semi_log.csv', separator=',', append=True)
callbacks.append(csv_logger)

# Fit the model
if isValid:
  model2.fit_generator(train_gen.flow(new_X, new_Y, batch_size=batch_size),
                    steps_per_epoch=10*new_X.shape[0]//batch_size,
                    epochs=epochs_2,
                    callbacks=callbacks,
                    validation_data=(X_valid, Y_valid))
else:
  model2.fit_generator(train_gen.flow(new_X, new_Y, batch_size=batch_size),
                    steps_per_epoch=10*new_X.shape[0]//batch_size,
                    epochs=epochs_2,
                    callbacks=callbacks)

# Save model
model.save(model_name)
