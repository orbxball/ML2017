import os
import sys
import argparse
import numpy as np
import pandas as pd
from CFModel import CFModel, DeepModel

def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('train', type=str)
    parser.add_argument('output', type=str)
    return parser.parse_args()


def predict_rating(trained_model, userid, movieid):
    return trained_model.rate(userid - 1, movieid - 1)


def ensure_dir(file_path):
  directory = os.path.dirname(file_path)
  if len(directory) == 0: return
  if not os.path.exists(directory):
    os.makedirs(directory)


def main(args):
    ratings = pd.read_csv(args.train, usecols=['UserID', 'MovieID', 'Rating'])
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    print('{} ratings loaded.'.format(ratings.shape[0]))

    users = pd.read_csv(USERS_CSV, sep='::', engine='python',
            usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    print('{} description of {} users loaded'.format(len(users), max_userid))

    movies = pd.read_csv(MOVIES_CSV, sep='::', engine='python',
            usecols=['movieID', 'Title', 'Genres'])
    print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

    test_data = pd.read_csv(TEST_CSV, usecols=['UserID', 'MovieID'])
    print('{} testing data loaded.'.format(test_data.shape[0]))

    trained_model = DeepModel(max_userid, max_movieid, DIM)
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

    TEST_CSV = 'data/test.csv'
    USERS_CSV = 'data/users.csv'
    MOVIES_CSV = 'data/movies.csv'
    MODEL_WEIGHTS_FILE = 'weights.h5'

    DIM = 120
    TEST_USER = 3000

    main(args)
