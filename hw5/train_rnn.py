import sys, os
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import load_model
from keras import backend as K

def read_data(file):
  print('Reading training data...')
  tags, texts, categories = [], [], []
  with open(file) as f:
    for line in f.readlines():
      buf = line.split('"')
      if len(buf) < 3: continue

      tags_tmp = buf[1].split(' ')
      for category in tags_tmp:
        categories.append(category)
      tags.append(tags_tmp)
      text = '"'.join(buf[2:])
      texts.append(text)

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(texts)
  index_seq = tokenizer.texts_to_sequences(texts)

  return tags, pad_sequences(index_seq), sorted(list(set(categories)))


def read_test(file):
  print('Reading test data...')
  texts = []
  with open(file) as f:
    for line in f.readlines():
      text = ','.join(line.split(',')[1:])
      if 'text' == text.strip(): continue
      texts.append(text)

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(texts)
  index_seq = tokenizer.texts_to_sequences(texts)
  return pad_sequences(index_seq)


def validate(X, Y, valid_size):
  permu = np.random.permutation(X.shape[0])
  x_valid = X[permu[:valid_size], :]
  y_valid = Y[permu[:valid_size], :]
  x_train = X[permu[valid_size:], :]
  y_train = Y[permu[valid_size:], :]
  return (x_train, y_train), (x_valid, y_valid)


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    """
    print('y_true: {}'.format(y_true))
    return fbeta_score(y_true, y_pred, beta=1)


def build_model(class_size, x_train, y_train, x_valid=None, y_valid=None, embedding_size=128):
  print('Build model...')
  model = Sequential()
  model.add(Embedding(max_vocab, embedding_size))
  model.add(LSTM(embedding_size, dropout=0.2, recurrent_dropout=0.2))
  model.add(Dense(class_size, activation='sigmoid'))

  # try using different optimizers and different optimizer configs
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[fmeasure])

  print('Train...')
  if is_valid:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=training_epoch,
              validation_data=(x_valid, y_valid))
  else:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=training_epoch,)

  return model


def main():
  if os.path.exists(model_name) and os.path.exists(categories_name):
    print('Loading model...')
    model = load_model(model_name, custom_objects={'fmeasure': fmeasure})
    categories = np.load(categories_name)
  else:
    tags, sequences, categories = read_data(train)
    print(len(tags), sequences.shape, len(categories))

    categorical_tags = np.zeros((len(tags), len(categories)))
    for i, tag in enumerate(tags):
      for item in tag:
        categorical_tags[i][categories.index(item)] = 1

    if is_valid:
      (x_train, y_train) , (x_valid, y_valid) = validate(sequences, categorical_tags, valid_size)
      print(x_train.shape, y_train.shape)
      print(x_valid.shape, y_valid.shape)
      model = build_model(len(categories), x_train, y_train,
                          x_valid=x_valid, y_valid=y_valid, embedding_size=128)
    else:
      x_train, y_train = sequences, categorical_tags
      model = build_model(len(categories), x_train, y_train,
                          embedding_size=256)

    print('Saving model...')
    model.save(model_name)
    np.save(categories_name, categories)

  x_test = read_test(test)
  pred = model.predict(x_test)
  pred[pred >= threshold] = 1
  pred[pred < threshold] = 0
  print(pred.shape)
  print(categories)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Homework 5: RNN')
  parser.add_argument('--train', metavar='<#train data path>',
                      type=str)
  parser.add_argument('--test', metavar='<#test data path>',
                      type=str, required=True)
  parser.add_argument('--output', metavar='<#output path>',
                      type=str, required=True)
  parser.add_argument('--valid', action='store_true')
  args = parser.parse_args()

  train = args.train
  test = args.test
  output = args.output
  is_valid = args.valid
  valid_size = 250
  training_epoch = 100
  batch_size = 32
  max_vocab = 60000
  threshold = 0.4
  model_name = 'rnn_model.h5'
  categories_name = 'rnn_categories.npy'

  main()