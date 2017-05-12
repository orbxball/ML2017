import sys, os
import argparse
import numpy as np
import gendata
from sklearn.neighbors import NearestNeighbors
from scipy import misc


def get_param():
  N = np.random.randint(1, 11) * 10000
  # the hidden dimension is randomly chosen from [60, 79] uniformly
  layer_dims = [np.random.randint(60, 80), 100]
  return N, layer_dims


def sampling(N, X):
  permu = np.random.permutation(N)[:sample_size]
  return X[permu,:]


def NN(fitting_data, query_data):
  nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', n_jobs=-1)
  return nbrs.fit(fitting_data).kneighbors(query_data)


def get_index(target, nparray):
  array = list(nparray)
  idx = 0
  for i in range(len(array)-1):
    if array[i] < target and target <= array[i+1]:
      idx = (target - array[i]) / (array[i+1] - array[i]) + i
      break
  if target >= array[len(array)-1]:
    idx = len(array)-1
  return idx


def train(sample_size, round_size, table_name):
  for dim in range(1, 61):
    average_distance = 0.0
    print('Dim: {}'.format(dim))
    for r in range(round_size):
      N, layer_dims = get_param()
      X = gendata.gen_data(dim, layer_dims, N)

      sample_X = sampling(N, X)
      distances, indices = NN(X, sample_X)

      avg_d = np.mean(distances[:,1])
      print('Round {}: Start NN with N={} hidden_dim={}, distance => {}'.format(r, N, layer_dims[0], avg_d))
      average_distance += avg_d
    average_distance /= round_size
    print('average distance => {}'.format(average_distance))
    table.append(average_distance)

  print('Saving table...')
  np.save(table_name, np.array(table))
  print('Saved')
  return table


def validate(model, sample_size):
  error = 0.0
  test_size = 200
  validate_round_size = 10
  for i in range(test_size):
    total_avg = 0.0
    dim = np.random.randint(1, 61)
    N, layer_dims = get_param()
    X = gendata.gen_data(dim, layer_dims, N)

    for j in range(validate_round_size):
      sample_X = sampling(N, X)
      distances, indices = NN(X, sample_X)

      avg_d = np.mean(distances[:,1])
      total_avg += avg_d
    predicted = get_index(total_avg/validate_round_size, model) + 1
    print('Validate {}: true => {} predicted => {}'.format(i, dim, predicted))
    error += np.absolute(np.log(predicted) - np.log(dim))
  error /= test_size
  print('Validation: {}'.format(error))


def main():
  try:
    print('Loading model...')
    model = np.load(model_name)
    print('Loading Done!')
  except:
    print('Start training...')
    table = train(sample_size, round_size, table_name)
    print('Training Done!')
    model = np.array(table)

  # validate(model, sample_size)
  # sys.exit(-1)

  if hand_dir != None:
    img_matrix = []
    for img in os.listdir(hand_dir):
      pic = misc.imread(os.path.join(hand_dir, img))
      new_pic = misc.imresize(pic, (10, 10))
      img_matrix.append(new_pic.flatten())
    img_matrix = np.array(img_matrix)
    std = np.std(img_matrix, axis=0)
    mean = np.mean(img_matrix, axis=0)
    img_matrix = (img_matrix - mean) / std
    sample_img_matrix = sampling(img_matrix.shape[0], img_matrix)

    distances, indices = NN(img_matrix, sample_img_matrix)
    avg_d = np.mean(distances[:,1])
    predicted = get_index(avg_d, model) + 1
    print(predicted)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='problem 3: Estimation of Intrinsic Diemnsion')
  parser.add_argument('--data', metavar='<#datapath>', type=str)
  parser.add_argument('--hand', metavar='<#hand path>', type=str, required=True)
  args = parser.parse_args()

  data_path = args.data
  hand_dir = args.hand

  # table record the average distance of dimension
  table = []
  model_dir = './model'
  table_name = os.path.join(model_dir, 'model2.npy')
  sample_size = 50
  round_size = 50
  model_name = os.path.join(model_dir, 'model2.npy')

  main()
