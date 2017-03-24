#!/usr/bin/env bash
if [ $# != 6 ]; then
	echo "There's no matching parameters!!!";
	exit -1;
fi

python3.6 best.py $3 $4 $5 $6
