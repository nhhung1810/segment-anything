#!/bin/bash
# "imp-aug-230606-002414"
# candidates=("imp-aug-230610-104249")
candidates=(
    "imp-aug-230610-104249"
    "imp-230603-150046"
    "imp-230610-011507"
    "imp-aug-230605-165716"
    "imp-aug-230605-000452"
    "imp-aug-230610-211354"
    # "imp-230608-220335"
    # "imp-aug-230606-002414"
    # "imp-230608-231031"
    # "imp-aug-230607-230424"
    # "imp-230601-213326"
    )
# for t in ${candidates[@]}; do
#     echo "runs/$t"
#     python bash_scripts/run_submission_mask_aug.py \
#         --model_dir "runs/$t" \
#         --prefix "thesis" \
#         --is_custom_class 1 \
#         --limit 20
# done

python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-aug-230610-104249" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 1 \
    --cuda 1 & \
python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-230603-150046" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 2 \
    --cuda 0 & \
python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-230610-011507" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 3 \
    --cuda 1 & \
python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-aug-230605-165716" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 4 \
    --cuda 0 & \
python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-aug-230605-000452" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 5 \
    --cuda 1 \
& python bash_scripts/run_submission_mask_aug.py \
    --model_dir "runs/imp-aug-230610-211354" \
    --prefix "thesis" \
    --is_custom_class 1 \
    --limit 20 \
    --pid 6 \
    --cuda 0