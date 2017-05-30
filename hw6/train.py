import os
import sys
import argparse
import numpy as np
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel, DeepModel

def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('train', type=str)
    parser.add_argument('test', type=str)
    return parser.parse_args()


def main(args):
    ratings = pd.read_csv(args.train,
                          usecols=['UserID', 'MovieID', 'Rating'])
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    ratings['User_emb_id'] = ratings['UserID'] - 1
    ratings['Movie_emb_id'] = ratings['MovieID'] - 1
    print('{} ratings loaded.'.format(ratings.shape[0]))

    ratings = ratings.sample(frac=1)
    Users = ratings['User_emb_id'].values
    print('Users: {}, shape = {}'.format(Users, Users.shape))
    Movies = ratings['Movie_emb_id'].values
    print('Movies: {}, shape = {}'.format(Movies, Movies.shape))
    Ratings = ratings['Rating'].values
    print('Ratings: {}, shape = {}'.format(Ratings, Ratings.shape))

    model = DeepModel(max_userid, max_movieid, DIM)
    model.compile(loss='mse', optimizer='adamax')

    callbacks = [EarlyStopping('val_loss', patience=2),
                 ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
    history = model.fit([Users, Movies], Ratings, epochs=100, validation_split=.1, verbose=1, callbacks=callbacks)


if __name__ == '__main__':
    args = parse_args()

    DIM = 120
    MODEL_WEIGHTS_FILE = 'weights.h5'

    main(args)
