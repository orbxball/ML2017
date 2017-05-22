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
  args = parser.parse_args()

  train_path = args.train
  test_path = args.test
  output_path = args.output

  base_dir = './model'
  max_features_size = 40000
  vectorizer_name = os.path.join(base_dir, 'vec')
  transformer_name = os.path.join(base_dir, 'trans')
  linear_svc_name = os.path.join(base_dir, 'linSVC')

  main()
