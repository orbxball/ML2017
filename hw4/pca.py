#!/usr/env/bin python
import sys, os
import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def read_pic(file, obj_size=10, pic_size=10):
  X = []
  base = 65
  global width, height
  pic = None
  for o in range(obj_size):
    for p in range(pic_size):
      pic_file = '{}{:02}.bmp'.format(chr(base+o), p)
      # print(pic_file)
      pic = misc.imread(os.path.join(pic_dir, pic_file))
      X.append(pic.flatten())
  width = pic.shape[0]
  height = pic.shape[1]
  return np.array(X)


def save_img(data, filename='default', subplot=False, size=0):
  global width, height

  print('Drawing {}...'.format(filename))
  if subplot:
    fig = plt.figure(figsize=(16, 16))
    for i in range(size):
      ax = fig.add_subplot(np.sqrt(size), np.sqrt(size), i+1)
      ax.imshow(data[i].reshape(width, height), cmap='gray')
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))
      plt.tight_layout()
      fig.savefig(filename)
  else:
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
    fig.savefig(filename)


def pca(x, face_size):
  global width, height
  mu = np.mean(x, axis=0)
  X = x - mu

  eigen_faces, sigma, v = np.linalg.svd(X.T, full_matrices=False)
  picked_faces = eigen_faces.T[:face_size]
  weights = np.dot(X, picked_faces.T)

  return mu.reshape(width, height), weights, picked_faces


def reconstruct(mu, weights, eigen_faces, eigen_size=0):
  pics = np.dot(weights, eigen_faces)
  pics += mu.flatten()
  save_img(pics, filename='original_eigen_{}.png'.format(eigen_size),
          subplot=True, size=pics.shape[0])

def cal_error(X, error):
  print('Start calculating...')
  for i in range(1, 101):
    mu, weights, eigen_faces = pca(X, i)
    pics = np.dot(weights, eigen_faces)
    rsme = np.sqrt(np.mean(np.square(X - mu.flatten() - pics))) / 256
    # print('Now {}, RSME: {}'.format(i, rsme))
    if rsme < error :
      return i


def main():
  pics_matrix = read_pic(pic_dir)

  mu, weights, eigen_faces = pca(pics_matrix, 9)

  # Save mean face
  #misc.imsave('average_face.png', mu)
  save_img(mu, filename='average_face.png')

  # Save eigen faces
  save_img(eigen_faces, filename='eigen_faces.png',
          subplot=True, size=eigen_faces.shape[0])

  # Save original image
  save_img(pics_matrix, filename='original.png',
          subplot=True, size=pics_matrix.shape[0])

  # Reconstrcut by eigen faces
  mu, weights, eigen_faces = pca(pics_matrix, 5)
  reconstruct(mu, weights, eigen_faces, eigen_size=eigen_faces.shape[0])

  # Calculate the RSME
  min_size_of_eigen_faces = cal_error(pics_matrix, 0.01)
  print('>>>{}<<<'.format(min_size_of_eigen_faces))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='problem 1: PCA')
  parser.add_argument('--data', metavar='<#data>', type=str, required=True)
  args = parser.parse_args()

  width = height = -1
  pic_dir = './' + args.data

  main()
