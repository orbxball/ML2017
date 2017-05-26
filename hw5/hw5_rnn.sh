#!/usr/bin/env bash

python3 glove_rnn.py --train model/model_raw --test $1 --output $2 --valid