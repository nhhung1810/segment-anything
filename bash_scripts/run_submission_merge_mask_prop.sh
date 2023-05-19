#!/bin/bash

# Full mask experiment
# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mask-prop-230508-222109" --limit 3 \
#     --is_custom_class 1

# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mask-prop-230509-005503" --limit 3 \
#     --is_custom_class 1

# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mask-prop-230511-153918" --limit 3 \
#     --is_custom_class 1

# Mask focus experiment, group 4 class + mask augmentation
# Run inference for top first 10 and top last 10
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230518-214209" \
    --is_custom_class 1
    # --limit -4 \
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230518-214209" \
    --is_custom_class 1
    # --limit 4 \

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230518-232718" \
    --is_custom_class 1
    # --limit -4 \

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230518-232718" \
    --is_custom_class 1
    # --limit 4 \

python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230519-012732" \
    --is_custom_class 1
    # --limit -4 \
python bash_scripts/run_submission_merge_mask_prop.py \
    --model_dir "runs/mask-drop-230519-012732" \
    --is_custom_class 1
    # --limit 4 \

# # Mask focus experiment, group 4 class
# # Run inference for top first 10 and top last 10
# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-111941" --limit -2 \
#     --is_custom_class 1
# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-111941" --limit 2 \
#     --is_custom_class 1

# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-131604" --limit -2
#     --is_custom_class 1
# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-131604" --limit 2 \
#     --is_custom_class 1

# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-151234" --limit -2 \
#     --is_custom_class 1
# python bash_scripts/run_submission_merge_mask_prop.py \
#     --model_dir "runs/mp-focus-230513-151234" --limit 2 \
#     --is_custom_class 1