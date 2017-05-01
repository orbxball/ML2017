#!/usr/bin/env python

import sys, os
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

def read_data(filename, height=48, width=48):
  try:
    print('Loading X.npy & Y.npy')
    X = np.load('X.npy')
    Y = np.load('Y.npy')
  except:
    with open(filename, "r+") as f:
      line = f.read().strip().replace(',', ' ').split('\n')[1:]
      raw_data = ' '.join(line)
      length = width*height+1 #1 is for label
      data = np.array(raw_data.split()).astype('float').reshape(-1, length)
      X = data[:, 1:]
      Y = data[:, 0]
      # Change data into CNN format
      X = X.reshape(-1, height, width, 1)
      Y = Y.reshape(-1, 1)
      print('Saving X.npy & Y.npy')
      np.save('X.npy', X) # (-1, height, width, 1)
      np.save('Y.npy', Y) # (-1, 1)
  return X, Y

def main():
  emotion_classifier = load_model(model_path)
  layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
  # print(layer_dict.keys())

  input_img = emotion_classifier.input
  name_ls = [name for name in layer_dict.keys() if 'leaky' in name]
  print(name_ls)
  collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

  X, Y = read_data(data_path)

  choose_id = 26000
  photo = X[choose_id].reshape(-1, height, width, 1)
  # print(photo.shape)

  for cnt, fn in enumerate(collect_layers):
    print('In the conv{}'.format(cnt))
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = min(im[0].shape[3], 32)
    for i in range(nb_filter):
      ax = fig.add_subplot(nb_filter/8, 8, i+1)
      print('imshow size:{}'.format(im[0][0, :, :, i].shape))
      ax.imshow(im[0][0, :, :, i], cmap='Blues')
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))
      plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    img_path = os.path.join(vis_dir, store_path)
    if not os.path.isdir(img_path):
      os.mkdir(img_path)
    fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='filters_output.py',
    description='ML-Assignment3 draw filters output.')
  parser.add_argument('--model', type=str, metavar='<#model>', required=True)
  parser.add_argument('--data', type=str, metavar='<#data>', required=True)

  args = parser.parse_args()
  model_path = args.model
  data_path = args.data

  height = width = 48

  base_dir = './'
  vis_dir = os.path.join(base_dir, 'output_vis')
  if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
  store_path = ''


  main()