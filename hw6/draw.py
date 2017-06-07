import os
import sys
import argparse
import numpy as np
import pandas as pd
from Model import build_cf_model, build_deep_model, rate
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

classes = ["Adventure|Western|Comedy", "Thriller|Horror|Mystery", "Crime|Film-Noir", "Sci-Fi|Fantasy", "Drama|Musical", "War|Documentary", "Children's|Animation", "Action|Romance"]

def parse_args():
    parser = argparse.ArgumentParser(description='HW6: drawing graph')
    parser.add_argument('data_dir', type=str)
    return parser.parse_args()

def draw(mapping, filename):
    print('Drawing...')
    fig = plt.figure(figsize=(10, 10), dpi=200)
    for i, key in enumerate(mapping.keys()):
        vis_x = mapping[key][:, 0]
        vis_y = mapping[key][:, 1]
        plt.scatter(vis_x, vis_y, marker='.', label=key)
    plt.xticks([])
    plt.yticks([])
    plt.legend(scatterpoints=1,
               loc='lower left',
               fontsize=8)
    plt.tight_layout()
    # plt.show()
    fig.savefig(filename)
    print('Done drawing!')

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main(args):
    users = pd.read_csv(USERS_CSV, sep='::', engine='python',
            usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    print('{} description of {} users loaded'.format(len(users), max_userid))

    movies = pd.read_csv(MOVIES_CSV, sep='::', engine='python',
            usecols=['movieID', 'Title', 'Genres'])
    print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

    test_data = pd.read_csv(TEST_CSV, usecols=['UserID', 'MovieID'])
    print('{} testing data loaded.'.format(test_data.shape[0]))

    trained_model = build_cf_model(max_userid, max_movieid, DIM, isBest=True)
    print('Loading model weights...')
    trained_model.load_weights(MODEL_WEIGHTS_FILE)
    print('Loading model done!!!')

    movies_array = movies.as_matrix()
    genres_map = {}
    for i in range(movies_array.shape[0]):
        genre = movies_array[i][2].split('|')[0]
        if genre not in genres_map.keys():
            genres_map[genre] = [movies_array[i][0] - 1]
        else:
            genres_map[genre].append(movies_array[i][0] - 1)
    # print(genres_map)
    movie_emb = np.array(trained_model.layers[3].get_weights()).squeeze()
    model = TSNE(n_components=2, random_state=0)
    movie_emb = model.fit_transform(movie_emb)
    for key in genres_map.keys():
        genres_map[key] = movie_emb[genres_map[key]]
        # print(key, genres_map[key].shape)

    new_genres_map = {}
    for c in classes:
        new_genres_map[c] = np.ndarray(shape=(0, 2))
        for g in c.split('|'):
            new_genres_map[c] = np.concatenate((new_genres_map[c], genres_map[g]), axis=0)
        # print(new_genres_map[c].shape)
    draw(new_genres_map, 'graph.png')


if __name__ == '__main__':
    args = parse_args()

    MODEL_DIR = './model'
    MAX_CSV = 'max_best.csv'
    TEST_CSV = 'test.csv'
    USERS_CSV = 'users.csv'
    MOVIES_CSV = 'movies.csv'
    MODEL_WEIGHTS_FILE = 'weights_add_const_dim15.h5'

    DATA_DIR = args.data_dir
    TEST_CSV = os.path.join(DATA_DIR, TEST_CSV)
    USERS_CSV = os.path.join(DATA_DIR, USERS_CSV)
    MOVIES_CSV = os.path.join(DATA_DIR, MOVIES_CSV)

    MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)
    MAX_CSV = os.path.join(MODEL_DIR, MAX_CSV)
    info = pd.read_csv(MAX_CSV)
    DIM = list(info['dim'])[0]
    max_userid = list(info['max_userid'])[0]
    max_movieid = list(info['max_movieid'])[0]

    main(args)
