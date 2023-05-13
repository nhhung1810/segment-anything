#!/bin/bash

for i in {1..13}
do
    python scripts/experiments/simple_mask_propagate/inference.py \
        --checkpoint="runs/mask-prop-230509-005503/model-100.pt" \
        --output_dir="runs/submission/mask-prop-230509-005503/class-$i/" \
        --class_num=$i && \
    python scripts/tools/evaluation/DSC_NSD_eval_fast.py \
        -g=dataset/FLARE22-version1/FLARE22_LabeledCase50/labels \
        -p="runs/submission/mask-prop-230509-005503/class-$i/" \
        --name="mp-005503-ck100-class-$i"
done