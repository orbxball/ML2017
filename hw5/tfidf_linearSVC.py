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
  all_corpus = texts + test_texts

  ### tokenize
  vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features_size)
  transformer = TfidfTransformer()
  transformer.fit(vectorizer.fit_transform(all_corpus))
  sequences = transformer.transform(vectorizer.transform(texts))
  test_data = transformer.transform(vectorizer.transform(test_texts))

  vectorizer2 = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=max_features_size, sublinear_tf=True)
  vectorizer2.fit(all_corpus)
  sequences2 = vectorizer2.transform(texts)
  test_data2 = vectorizer2.transform(test_texts)

  if is_valid:
    (x_train, y_train),(x_valid, y_valid) = validate(sequences, tags, valid_size)
    (x_train2, y_train2),(x_valid2, y_valid2) = validate(sequences2, tags, valid_size)
  else:
    x_train, y_train = sequences, tags
    x_train2, y_train2 = sequences2, tags

  linear_svc = OneVsRestClassifier(LinearSVC(C=5e-4, class_weight='balanced'))
  linear_svc2 = OneVsRestClassifier(LinearSVC(C=5e-4, class_weight='balanced'))

  ### cross validation
  scores = cross_val_score(linear_svc, x_train, y_train, cv=8, scoring='f1_samples', n_jobs=-1)
  print(scores, scores.mean(), scores.std())
  scores2 = cross_val_score(linear_svc2, x_train2, y_train2, cv=8, scoring='f1_samples', n_jobs=-1)
  print(scores2, scores2.mean(), scores2.std())

  ### predict
  linear_svc.fit(x_train, y_train)
  predict = linear_svc.predict(test_data)
  linear_svc2.fit(x_train2, y_train2)
  predict2 = linear_svc2.predict(test_data2)
  # print(mlb.classes_)

  ### save vectorizer + transformer + linearSVC
  with open(vectorizer_name, 'wb') as f:
    pickle.dump(vectorizer, f)
  with open(transformer_name, 'wb') as f:
    pickle.dump(transformer, f)
  with open(linear_svc_name, 'wb') as f:
    pickle.dump(linear_svc, f)

  with open(vectorizer_name2, 'wb') as f:
    pickle.dump(vectorizer2, f)
  with open(linear_svc_name2, 'wb') as f:
    pickle.dump(linear_svc2, f)

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
  max_features_size = 40000
  vectorizer_name = 'vec'
  transformer_name = 'trans'
  linear_svc_name = 'linSVC'
  vectorizer_name2 = 'vec2'
  linear_svc_name2 = 'linSVC2'

  main()
