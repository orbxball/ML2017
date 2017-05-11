import sys, os
import argparse
import numpy as np
import gendata
from sklearn.neighbors import NearestNeighbors


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


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


def test(model, test_data):
  ans = []
  test_round_size = 10
  total_avg = 0.0
  for i in test_data.keys():
    print('Predicting {}...'.format(i))
    for j in range(test_round_size):
      sample_test_data = sampling(test_data[i].shape[0], test_data[i])
      distances, indices = NN(test_data[i], sample_test_data)

      avg_d = np.mean(distances[:,1])
      total_avg += avg_d
      print('Round: {}, avg_d: {}'.format(j, avg_d))
    total_avg /= test_round_size
    predicted = get_index(total_avg, model) + 1

    ans.append(np.log(predicted))
  return ans


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

  ans = test(model, np.load(data_path))

  ensure_dir(output_path)
  result = []
  for index, value in enumerate(ans):
    result.append("{0},{1}".format(index, value))
  with open(output_path, "w+") as f:
    f.write("SetId,LogDim\n")
    f.write("\n".join(result))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='problem 3: Estimation of Intrinsic Diemnsion')
  parser.add_argument('--data', metavar='<#datapath>', type=str, required=True)
  parser.add_argument('--out', metavar='<#output>', type=str, required=True)
  args = parser.parse_args()

  data_path = args.data
  output_path = args.out

  # table record the average distance of dimension
  table = []
  model_dir = './model'
  table_name = os.path.join(model_dir, 'model2.npy')
  sample_size = 50
  round_size = 50
  model_name = os.path.join(model_dir, 'model2.npy')

  main()
