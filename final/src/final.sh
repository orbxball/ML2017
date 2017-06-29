#!/usr/bin/env bash

if [ $# != 3 ]; then
    echo "Usage: bash final.sh [training set values] [training set labels] [testing set values]";
fi

wget www.csie.ntu.edu.tw/~b03502040/8275.zip
unzip 8275.zip

python3 train.py --train $1 --label $2 --test $3  --model depth23/*
