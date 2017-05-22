#!/usr/bin/env bash

python3 glove_rnn.py --train data/train_data.csv --test $1 --output $2 --valid