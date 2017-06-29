import os
import logging
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import mode
from utils import DataProcessor


def parse_args():
    parser = argparse.ArgumentParser('Pump it up.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--cv', type=int, default=4)
    parser.add_argument('--eta', type=float, default=0.025)
    parser.add_argument('--depth', type=int, default=23)
    parser.add_argument('--seed', nargs=2, type=int, default=[60, 73])
    parser.add_argument('--model', nargs='*')
    return parser.parse_args()


def main(args):

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    data = DataProcessor()
    
    logger.info('Read csvs')
    data.read_data(args.train, args.test, args.label)

    logger.info('Preprocess data')
    data.preprocess()
    
    train_dmatrix = xgb.DMatrix(data=data.train, label=data.labels, missing=np.nan)
    test_dmatrix = xgb.DMatrix(data=data.test, missing=np.nan)

    if args.model is None:
        depth_dir = 'depth{}'.format(args.depth)
        if not os.path.exists(depth_dir):
            os.mkdir(depth_dir)
        
        param = {
                'booster': 'gbtree',
                'obective': 'multi:softmax',
                'eta': args.eta,
                'max_depth': args.depth,
                'colsample_bytree': 0.4,
                'silent': 1,
                'eval_metric': 'merror',
                'num_class': 4
                }

        logger.info('Start training from seed {} to {}'.format(args.seed[0], args.seed[1]-1))
        for i in range(args.seed[0], args.seed[1]):
            logger.info('Cross validate with seed {}, depth {}, {}-fold'.format(i, args.depth, args.cv))
            
            param['seed'] = i
            #res = xgb.cv(param, dtrain=train_dmatrix, seed=i, num_boost_round=500, 
            #        nfold=args.cv, early_stopping_rounds=30, maximize=False, verbose_eval=True)
            #num_boost_round = res['test-merror-mean'].argmin()
            num_boost_round = 210
            logger.info('Train xgboost tree with seed {}, depth {}, num_boost_round {}'.format(i, args.depth, num_boost_round))
            
            clf = xgb.train(param, dtrain=train_dmatrix, num_boost_round=num_boost_round, maximize=False)
            
            save_path = os.path.join(depth_dir, 'xgb-model-seed-{}'.format(i))
            
            clf.save_model(save_path)

            logger.info('Save xgboost tree at {}'.format(save_path))

        logger.info('End of training. All models are saved at {}'.format(depth_dir))

    else:
        pred_overall = []
        for mfile in args.model:
            logger.info('Load xgboost tree model {}'.format(mfile))
            clf = xgb.Booster()
            clf.load_model(mfile)

            pred = clf.predict(data=test_dmatrix).astype(int)
            pred_overall.append(pred)
        
        pred_overall = mode(pred_overall, axis=0)[0].squeeze()
        data.write_data('output.csv', pred_overall)


if __name__ == '__main__':
    args = parse_args()
    main(args)
