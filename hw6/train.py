import os
import sys
import argparse
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from Model import build_cf_model, build_deep_model, rate


def parse_args():
    parser = argparse.ArgumentParser(description='HW6: Matrix Factorization')
    parser.add_argument('train', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('--dim', type=int, default=15)
    return parser.parse_args()


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

    model = build_cf_model(max_userid, max_movieid, DIM)
    model.compile(loss='mse', optimizer='adamax', metrics=[rmse])

    callbacks = [EarlyStopping('val_rmse', patience=2),
                 ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
    history = model.fit([Users, Movies], Ratings, epochs=1000, batch_size=256, validation_split=.1, verbose=1, callbacks=callbacks)


if __name__ == '__main__':
    args = parse_args()

    MODEL_DIR = './model'
    DIM = args.dim
    MODEL_WEIGHTS_FILE = 'weights.h5'
    MAX_FILE = 'max.csv'

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    MODEL_WEIGHTS_FILE = os.path.join(MODEL_DIR, MODEL_WEIGHTS_FILE)
    MAX_FILE = os.path.join(MODEL_DIR, MAX_FILE)
    main(args)
