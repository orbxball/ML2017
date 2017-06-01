import os
import sys
import argparse
import numpy as np
import pandas as pd
from Model import build_cf_model, build_deep_model, rate


def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def predict_rating(trained_model, userid, movieid):
    return rate(trained_model, userid - 1, movieid - 1)


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

    trained_model = build_cf_model(max_userid, max_movieid, DIM)
    print('Loading model weights...')
    trained_model.load_weights(MODEL_WEIGHTS_FILE)
    print('Loading model done!!!')

    recommendations = pd.read_csv(TEST_CSV, usecols=['TestDataID'])
    recommendations['Rating'] = test_data.apply(lambda x: predict_rating(trained_model, x['UserID'], x['MovieID']), axis=1)
    # print(recommendations)

    ensure_dir(args.output)
    recommendations.to_csv(args.output, index=False, columns=['TestDataID', 'Rating'])


if __name__ == '__main__':
    args = parse_args()

    MODEL_DIR = './model'
    MAX_CSV = 'max.csv'
    TEST_CSV = 'test.csv'
    USERS_CSV = 'users.csv'
    MOVIES_CSV = 'movies.csv'
    MODEL_WEIGHTS_FILE = 'weights.h5'

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
