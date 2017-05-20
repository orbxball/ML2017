import sys, os
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score


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


def validate(X, Y, valid_size):
  x_train = X[:valid_size, :]
  y_train = Y[:valid_size, :]
  x_valid = X[valid_size:, :]
  y_valid = Y[valid_size:, :]
  return (x_train, y_train), (x_valid, y_valid)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main():
  ### read training data & testing data
  tags, texts, mlb = read_data(train_path)
  test_texts = read_test(test_path)

  ### Tokenize
  vectorizer = TfidfVectorizer(stop_words='english')
  sequences = vectorizer.fit_transform(texts)
  test_data = vectorizer.transform(test_texts)

  (x_train, y_train),(x_valid, y_valid) = validate(sequences, tags, valid_size)
  print(x_train.shape)
  print(y_train.shape)
  print(x_valid.shape)
  print(y_valid.shape)

  linear_svc = OneVsRestClassifier(LinearSVC(C=1e-3, class_weight='balanced'))
  linear_svc.fit(x_train, y_train)
  y_train_predict = linear_svc.predict(x_train)
  y_valid_predict = linear_svc.predict(x_valid)
  print(f1_score(y_valid, y_valid_predict, average='micro'))
  predict = linear_svc.predict(test_data)
  # print(mlb.classes_)

  # Test data
  ensure_dir(output_path)
  result = []
  for i, categories in enumerate(mlb.inverse_transform(predict)):
    ret = []
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
  parser.add_argument('--valid', action='store_true')
  args = parser.parse_args()

  train_path = args.train
  test_path = args.test
  output_path = args.output
  is_valid = args.valid
  valid_size = -400
  max_vocab = 60000

  main()