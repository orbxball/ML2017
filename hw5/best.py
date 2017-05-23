import sys, os
import argparse
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences


def read_data(file):
  print('Reading training data...')
  tags, texts = [], []
  with open(file) as f:
    f.readline()
    for line in f:
      buf = line.split('"', 2)

      tags_tmp = buf[1].split(' ')
      tags.append(tags_tmp)
      text = buf[2][1:]
      texts.append(text)

  mlb = MultiLabelBinarizer()
  tags = mlb.fit_transform(tags)
  print('Classes Number: {}'.format(len(mlb.classes_)))
  return tags, texts, mlb


def read_test(file):
  print('Reading test data...')
  texts = []
  with open(file) as f:
    next(f)
    for line in f:
      text = ','.join(line.split(',')[1:])
      texts.append(text)
  return texts


def preprocess(train, test, tokenizer, maxlen=400, split=" ", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'):
  sequences_train = tokenizer.texts_to_sequences(train)
  sequences_test = tokenizer.texts_to_sequences(test)
  return pad_sequences(sequences_train, maxlen=maxlen), pad_sequences(sequences_test, maxlen=maxlen)


def fmeasure(y_true, y_pred):
    thres = 0.4
    y_pred = K.cast(K.greater(y_pred, thres), dtype='float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return K.mean(f)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main():
  ### read training data & testing data
  tags, texts, mlb = read_data(train_path)
  test_texts = read_test(test_path)
  all_corpus = texts + test_texts


  ### load the vectorizer + transformer + linerSVC
  with open(vectorizer_name, 'rb') as f:
    vectorizer = pickle.load(f)
  with open(transformer_name, 'rb') as f:
    transformer = pickle.load(f)
  with open(linear_svc_name, 'rb') as f:
    linear_svc = pickle.load(f)
  sequences = transformer.transform(vectorizer.transform(texts))
  test_data = transformer.transform(vectorizer.transform(test_texts))
  x_train, y_train = sequences, tags

  ### cross validation
  scores = cross_val_score(linear_svc, x_train, y_train, cv=8, scoring='f1_samples', n_jobs=-1)
  print(scores, scores.mean(), scores.std())

  ### predict
  predict = linear_svc.predict(test_data)
  # print(mlb.classes_)


  ### load tfidf vectorizer + linerSVC
  with open(vectorizer_name2, 'rb') as f:
    vectorizer = pickle.load(f)
  with open(linear_svc_name2, 'rb') as f:
    linear_svc = pickle.load(f)
  sequences = vectorizer.transform(texts)
  test_data = vectorizer.transform(test_texts)
  x_train, y_train = sequences, tags

  ### cross validation
  scores = cross_val_score(linear_svc, x_train, y_train, cv=8, scoring='f1_samples', n_jobs=-1)
  print(scores, scores.mean(), scores.std())

  ### predict
  predict2 = linear_svc.predict(test_data)


  ### RNN 1
  print('load tokenizer')
  tokenizer = pickle.load(open(tokenizer_name, 'rb'))
  sequences, sequences_test = preprocess(texts, test_texts, tokenizer)
  model = load_model(model_name, custom_objects={'fmeasure': fmeasure})
  predict3 = model.predict(sequences_test)
  predict3[predict3 < threshold] = 0
  predict3[predict3 >= threshold] = 1


  ### Voting
  pred = predict + predict2 + predict3
  pred[pred < 1.5] = 0
  pred[pred >= 1.5] = 1


  # Test data
  ensure_dir(output_path)
  result = []
  mlb_backup = mlb.inverse_transform(predict3)
  for i, categories in enumerate(mlb.inverse_transform(pred)):
    ret = []
    if len(categories) == 0:
      categories = mlb_backup[i]
    for category in categories:
      ret.append(category)
    result.append('"{0}","{1}"'.format(i, " ".join(ret)))
  with open(output_path, "w+") as f:
    f.write('"id","tags"\n')
    f.write("\n".join(result))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Homework 5: WindQAQ')
  parser.add_argument('--train', metavar='<#train data path>',
                      type=str, required=True)
  parser.add_argument('--test', metavar='<#test data path>',
                      type=str, required=True)
  parser.add_argument('--output', metavar='<#output path>',
                      type=str, required=True)
  args = parser.parse_args()

  train_path = args.train
  test_path = args.test
  output_path = args.output

  base_dir = './model'
  max_features_size = 40000

  vectorizer_name = os.path.join(base_dir, 'vec')
  transformer_name = os.path.join(base_dir, 'trans')
  linear_svc_name = os.path.join(base_dir, 'linSVC')
  vectorizer_name2 = os.path.join(base_dir, 'vec2')
  linear_svc_name2 = os.path.join(base_dir, 'linSVC2')
  tokenizer_name = os.path.join(base_dir, 'word_index')
  model_name = os.path.join(base_dir, 'model-5.h5')
  threshold = 0.4

  main()
