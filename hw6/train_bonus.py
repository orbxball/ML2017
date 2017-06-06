import os
import sys
import argparse
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from Model import build_cf_model, build_deep_model, rate

classes = ["Adventure", "Western", "Comedy", "Thriller", "Horror", "Mystery", "Crime", "Film-Noir", "Sci-Fi", "Fantasy", "Drama", "Musical", "War", "Documentary", "Children's", "Animation", "Action", "Romance"]


def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('train', type=str)
    parser.add_argument('users', type=str)
    parser.add_argument('movies', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('--dim', type=int, default=15)
    return parser.parse_args()

def make_users(row, matrix):
    matrix[row['UserID']] = [row['UserID'], row['Gender'], row['Age'], row['Occupation']]
    return row

def categorize_movie(row, matrix, idx_map):
    x = [0] * len(classes)
    for g in row['Genres'].split('|'):
        x[idx_map[g]] = 1
    matrix[row['movieID']] = [row['movieID']] + x

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1., 5.)
    return K.sqrt(K.mean(K.square((y_true - y_pred))))

def main(args):
    ratings = pd.read_csv(args.train,
                          usecols=['UserID', 'MovieID', 'Rating'])
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    ratings['User_emb_id'] = ratings['UserID'] - 1
    ratings['Movie_emb_id'] = ratings['MovieID'] - 1
    print('{} ratings loaded.'.format(ratings.shape[0]))

    users = pd.read_csv(args.users, sep='::', engine='python',
            usecols=['UserID', 'Gender', 'Age', 'Occupation'])
    users['UserID'] -= 1
    users['Gender'][users['Gender'] == 'F'] = 0
    users['Gender'][users['Gender'] == 'M'] = 1
    users_mx = {}
    users.apply(lambda x: make_users(x, users_mx), axis=1)
    print('{} description of {} users loaded'.format(len(users), max_userid))

    movies = pd.read_csv(args.movies, sep='::', engine='python',
            usecols=['movieID', 'Genres'])
    movies['movieID'] -= 1
    movies_mx = {}
    classes_idx = {}
    for i, c in enumerate(classes):
        classes_idx[c] = i
    movies.apply(lambda x: categorize_movie(x, movies_mx, classes_idx), axis=1)
    print('{} descriptions of {} movies loaded'.format(len(movies), max_movieid))

    maximum = {}
    maximum['max_userid'] = [max_userid]
    maximum['max_movieid'] = [max_movieid]
    maximum['dim'] = [DIM]
    pd.DataFrame(data=maximum).to_csv(MAX_FILE, index=False)
    print('max info save to {}'.format(MAX_FILE))

    ratings = ratings.sample(frac=1)
    Users = ratings['User_emb_id'].values
    print('Users: {}, shape = {}'.format(Users, Users.shape))
    Movies = ratings['Movie_emb_id'].values
    print('Movies: {}, shape = {}'.format(Movies, Movies.shape))
    Ratings = ratings['Rating'].values
    print('Ratings: {}, shape = {}'.format(Ratings, Ratings.shape))

    new_Users = np.array(list(map(users_mx.get, Users)))
    new_Movies = np.array(list(map(movies_mx.get, Movies)))

    model = build_deep_model(max_userid, max_movieid, DIM)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])

    callbacks = [EarlyStopping('val_rmse', patience=2),
                 ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
    history = model.fit([new_Users, new_Movies], Ratings, epochs=1000, batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)


if __name__ == '__main__':
    args = parse_args()

    MODEL_DIR = './model'
    DIM = args.dim
    MODEL_WEIGHTS_FILE = 'weights_bonus.h5'
    MAX_FILE = 'max_bonus.csv'

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)
    MAX_FILE = os.path.join(MODEL_DIR, MAX_FILE)
    main(args)
