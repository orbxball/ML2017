#!/usr/bin/env python
import sys, os
import itertools
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import confusion_matrix

# Parameter
height = width = 48
num_classes = 7
model_name = sys.argv[2]
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
option = sys.argv[3]  # draw: draw confusion mastrix
                      # pick: pick a image fitting the type II error

# Read the train data
print('Read start')
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
print('Read finished!')

# Split the data
valid_num = 3000
X_train, Y_train = X[:-valid_num], Y[:-valid_num].astype('int')
X_valid, Y_valid = X[-valid_num:], Y[-valid_num:].astype('int')
# print(X_train.shape)
# print(X_valid.shape)
# print(Y_train.shape)
# print(Y_valid.shape)

# Load model
model = load_model(model_name)
print('Predicting')
pred = model.predict(X_valid)
# print(pred.shape)
pred_label = np.argmax(pred, axis=1)
# print(pred_label.shape)
Y_valid = Y_valid.reshape(-1, )
# print(Y_valid.shape)
print('Predicting done!')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.3f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="brown" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
if option == 'draw':
  cnf_matrix = confusion_matrix(Y_valid, pred_label)
  np.set_printoptions(precision=3)

  # Plot non-normalized confusion matrix
  # plt.figure()
  # plot_confusion_matrix(cnf_matrix, classes=class_names,
  #                       title='Confusion matrix, without normalization')

  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')

  plt.show()

# Pick image
elif option == 'pick':
  base_dir = './'
  img_dir = os.path.join(base_dir, 'cm_image')
  if not os.path.exists(img_dir):
    os.makedirs(img_dir)

  true_label = np.argwhere(Y_valid == 3).squeeze()
  # print(true_label)
  picked_label = np.argwhere(pred_label[true_label] == 3).squeeze()
  # print(picked_label)

  idx = true_label[picked_label[3]]
  print('Picking image number {}'.format(idx))
  see = X_valid[idx].reshape(height, width)
  # print(see.shape)
  ans = ['{:.3f}'.format(i) for i in list(pred[idx])]
  print('True label: {:d}; Predicted label: {}'.format(Y_valid[idx], pred_label[idx]))
  print('Its percentage: {}'.format(' , '.join(ans)))

  plt.figure()
  plt.imshow(see,cmap='gray')
  plt.colorbar()
  plt.tight_layout()
  fig = plt.gcf()
  plt.draw()
  fig.savefig(os.path.join(img_dir, '{}.png'.format(idx)), dpi=100)