#!/bin/bash

# Full mask experiment
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-prop-230508-222109" --limit 3 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-prop-230509-005503" --limit 3 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-prop-230511-153918" --limit 3 \
    --is_custom_class 1

# Mask focus experiment, group 2 class
# Run inference for top first 10 and top last 10
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230512-221332" --limit -2 \
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230512-221332" --limit 2 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-014610" --limit -2 \
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-014610" --limit 2 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-023320" --limit -2 \
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-023320" --limit 2 \
    --is_custom_class 1

# Mask focus experiment, group 4 class
# Run inference for top first 10 and top last 10
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-111941" --limit -2 \
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-111941" --limit 2 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-131604" --limit -2
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-131604" --limit 2 \
    --is_custom_class 1

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-151234" --limit -2 \
    --is_custom_class 1
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mp-focus-230513-151234" --limit 2 \
    --is_custom_class 1