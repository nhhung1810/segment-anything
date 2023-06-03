#!/bin/bash

python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-230601-213326" \
    --is_custom_class 1 \
    --prefix "imp"
    # --skip_eval 1 \
    # --limit 5 \

# python bash_scripts/run_submission_mask_aug.py \
#     --model_dir "runs/mask-aug-230521-035709" \
#     --is_custom_class 1 \
#     --prefix "maug_19" \
#     --skip_eval 1 \
#     --limit 5 \

# python bash_scripts/run_submission_mask_aug.py \
#     --model_dir "runs/mask-aug-230520-140411" \
#     --prefix "maug_all" \
#     --limit 6 \
#     --skip_eval 1
    