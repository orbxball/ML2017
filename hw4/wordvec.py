#!/usr/env/bin python
import sys, os
import argparse
import numpy as np
import word2vec
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

def word_training(path, embedded_size):
  dirname = os.path.dirname(path)
  filename = os.path.basename(path)
  phrasesname = os.path.join(dirname, '{}-phrases'.format(filename))
  modelname = os.path.join(dirname, '{}.bin'.format(filename))
  print('Training...')
  word2vec.word2phrase(path, phrasesname)
  word2vec.word2vec(phrasesname, modelname, size=embedded_size)
  print('Training Done!!!')
  return modelname

def POS_tag(model, size=0):
  tags = ['JJ', 'NNP', 'NN', 'NNS']
  punctuations = [',', '.', ':', ';', '!', '?', '“', '”', '’']

  # usage: nltk.pos_tag([vocab])
  filter_vocabs = []
  filter_idx = []
  for i, vocab in enumerate(model.vocab[:size]):
    if len(vocab) > 1 and not any(punc in vocab for punc in punctuations) and nltk.pos_tag([vocab])[0][1] in tags:
      filter_vocabs.append(vocab)
      filter_idx.append(i)
  return filter_idx, filter_vocabs

def tsne(data, size=0):
  model = TSNE(n_components=2, random_state=0)
  return model.fit_transform(data[:size])


def plot(vectors, vocabs, filename='default.png'):
  # print(vectors.shape)
  # print(len(vocabs))
  xs, ys = vectors[:,0], vectors[:, 1]
  xs *= 10000
  ys *= 10000
  texts = []
  fig, ax = plt.subplots(figsize=(16, 16))
  for x, y, vocab in zip(xs, ys, vocabs):
    ax.plot(x, y, '.')
    texts.append(plt.text(x, y, vocab))
  adjust_text(texts, arrowprops=dict(arrowstyle='-'))
  plt.xticks(np.array([]))
  plt.yticks(np.array([]))
  fig.savefig(filename)

def main():
  embedded_size = 500
  k = 400
  print('Embedded size: {}, Number of picked: {}'.format(embedded_size, k))

  model_path = word_training(data_path, embedded_size)
  print('Loading model...')
  model = word2vec.load(model_path)
  print('Loading Done!!!')

  plane = tsne(model.vectors, size=k)
  indexs, vocabs = POS_tag(model, size=k)
  plot(plane[indexs], vocabs, 'word2vec_{}_{}.png'.format(embedded_size, k))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='problem 2: word2vec + NLTK')
  parser.add_argument('--data', metavar='<data path>', type=str, required=True)
  args = parser.parse_args()

  data_path = args.data

  main()