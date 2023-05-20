#!/bin/bash

python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/mask-aug-230520-140411" \
    --is_custom_class 1 \
    --prefix "maug_19" \
    --limit 8

# python bash_scripts/run_submission_mask_aug.py \
#     --model_dir "runs/mask-aug-230520-140411" \
#     --prefix "maug_all" \
#     --limit 6 \
#     --skip_eval 1
    