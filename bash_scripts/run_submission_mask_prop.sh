#!/bin/bash

for i in {1..13}
do
    python scripts/experiments/simple_mask_propagate/inference.py \
        --class_num=$i
done