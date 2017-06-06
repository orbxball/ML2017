import os
import sys
import argparse
import numpy as np
import pandas as pd
from Model import build_cf_model, build_deep_model, rate

classes = ["Adventure", "Western", "Comedy", "Thriller", "Horror", "Mystery", "Crime", "Film-Noir", "Sci-Fi", "Fantasy", "Drama", "Musical", "War", "Documentary", "Children's", "Animation", "Action", "Romance"]


def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()

def make_users(row, matrix):
    matrix[row['UserID']] = [row['UserID'], row['Gender'], row['Age'], row['Occupation']]
    return row

def categorize_movie(row, matrix, idx_map):
    x = [0] * len(classes)
    for g in row['Genres'].split('|'):
        x[idx_map[g]] = 1
    matrix[row['movieID']] = [row['movieID']] + x

def predict_rating(trained_model, userid, movieid):
    return rate(trained_model, userid, movieid)

def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main(args):
    users = pd.read_csv(USERS_CSV, sep='::', engine='python',
            usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    users['UserID'] -= 1
    users['Gender'][users['Gender'] == 'F'] = 0
    users['Gender'][users['Gender'] == 'M'] = 1
    users_mx = {}
    users.apply(lambda x: make_users(x, users_mx), axis=1)
    print('{} description of {} users loaded'.format(len(users), max_userid))

    movies = pd.read_csv(MOVIES_CSV, sep='::', engine='python',
            usecols=['movieID', 'Title', 'Genres'])
    movies['movieID'] -= 1
    movies_mx = {}
    classes_idx = {}
    for i, c in enumerate(classes):
        classes_idx[c] = i
    movies.apply(lambda x: categorize_movie(x, movies_mx, classes_idx), axis=1)
    print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

    test_data = pd.read_csv(TEST_CSV, usecols=['UserID', 'MovieID'])
    print('{} testing data loaded.'.format(test_data.shape[0]))

    trained_model = build_deep_model(max_userid, max_movieid, DIM)
    print('Loading model weights...')
    trained_model.load_weights(MODEL_WEIGHTS_FILE)
    print('Loading model done!!!')

    recommendations = pd.read_csv(TEST_CSV, usecols=['TestDataID'])
    recommendations['Rating'] = test_data.apply(lambda x: predict_rating(trained_model, users_mx[x['UserID']-1], movies_mx[x['MovieID']-1]), axis=1)
    # print(recommendations)

    ensure_dir(args.output)
    recommendations.to_csv(args.output, index=False, columns=['TestDataID', 'Rating'])


if __name__ == '__main__':
    args = parse_args()

    MODEL_DIR = './model'
    MAX_CSV = 'max_bonus.csv'
    TEST_CSV = 'test.csv'
    USERS_CSV = 'users.csv'
    MOVIES_CSV = 'movies.csv'
    MODEL_WEIGHTS_FILE = 'weights_bonus.h5'

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
