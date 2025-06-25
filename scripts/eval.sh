#!/bin/bash

# Ex: ./scripts/eval.sh 0.1 model_path
PROB=$1
MODEL_PATH=$2
python3 eval.py $PROB $MODEL_PATH
