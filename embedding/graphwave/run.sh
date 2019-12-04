#!/bin/bash
FILES=../../input/*
DIMENSION=2
for f in $FILES
do
    filename=${f#*../../input/}
    filename=${filename%.*}_${DIMENSION}d.emb
    python src/main.py --edgelist-input --input $f --output 'output/'$filename --sample-number $DIMENSION
done