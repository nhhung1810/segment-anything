#!/bin/bash

for i in {1..13}
do
    python scripts/experiments/one_point/inference.py \
        --checkpoint="runs/sam-one-point-230501-152140/model-40.pt" \
        --output_dir="runs/submission/sam-one-point-230501-152140/class-$i/" \
        --class_num=$i && \
    python scripts/tools/evaluation/DSC_NSD_eval_fast.py \
        -g=dataset/FLARE22-version1/FLARE22_LabeledCase50/labels \
        -p="runs/submission/sam-one-point-230501-152140/class-$i/" \
        --name="op-152140-ck40-class-$i"
done