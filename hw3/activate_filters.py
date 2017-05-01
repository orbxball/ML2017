#!/usr/bin/env python

import sys, os
import argparse
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x

def normalize(x):
  # utility function to normalize a tensor by its L2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
  """
  Implement this function!
  """
  filter_images = []
  step = 1e-2
  for i in range(num_step):
    loss_value, grads_value = iter_func([input_image_data, 0])
    input_image_data += grads_value * step
    if i % RECORD_FREQ == 0:
      filter_images.append((input_image_data, loss_value))
      print('#{}, loss rate: {}'.format(i, loss_value))
  return filter_images

def main():
  emotion_classifier = load_model(model_name)
  layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)
  input_img = emotion_classifier.input

  name_ls = [name for name in layer_dict.keys() if 'leaky' in name]
  collect_layers = [ layer_dict[name].output for name in name_ls ]

  for cnt, c in enumerate(collect_layers):
    filter_imgs = []
    for filter_idx in range(nb_filter):
      input_img_data = np.random.random((1, 48, 48, 1)) # random noise
      target = K.mean(c[:, :, :, filter_idx])
      grads = normalize(K.gradients(target, input_img)[0])
      iterate = K.function([input_img, K.learning_phase()], [target, grads])

      ###
      "You need to implement it."
      print('==={}==='.format(filter_idx))
      filter_imgs.append(grad_ascent(num_step, input_img_data, iterate))
      ###
    print('Finish gradient')

    for it in range(NUM_STEPS//RECORD_FREQ):
      print('In the #{}'.format(it))
      fig = plt.figure(figsize=(14, 8))
      for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/8, 8, i+1)
        raw_img = filter_imgs[i][it][0].squeeze()
        ax.imshow(deprocess_image(raw_img), cmap='Blues')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('{:.3f}'.format(filter_imgs[i][it][1]))
        plt.tight_layout()
      # fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
      img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
      if not os.path.exists(img_path):
        os.mkdir(img_path)
      fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog='activate_filters.py',
    description='ML-Assignment3 activate filters.')
  parser.add_argument('--model', type=str, metavar='<#model>', required=True)

  args = parser.parse_args()
  model_name = args.model

  num_step = NUM_STEPS = 100
  RECORD_FREQ = 10
  nb_filter = 32

  base_dir = './'
  filter_dir = os.path.join(base_dir, 'filter_vis')
  if not os.path.exists(filter_dir):
    os.mkdir(filter_dir)
  store_path = ''

  main()
