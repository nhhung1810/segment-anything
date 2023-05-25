#!/bin/bash

# Mask augmentation

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-aug-230520-140411" \
    --is_custom_class 1 \
    --prefix "bi_maug"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-aug-230521-004441" \
    --is_custom_class 1 \
    --prefix "bi_maug"
    # --limit 5 \

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-aug-230521-035709" \
    --is_custom_class 1 \
    --prefix "bi_maug"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-aug-230521-210435" \
    --is_custom_class 1 \
    --prefix "bi_maug"
    
# Mask drop

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-liver-first-augment/mask-drop-230518-214209" \
    --is_custom_class 1 \
    --prefix "bi_drop"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-liver-first-augment/mask-drop-230518-232718" \
    --is_custom_class 1 \
    --prefix "bi_drop"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-liver-first-augment/mask-drop-230519-012732" \
    --is_custom_class 1 \
    --prefix "bi_drop"

# mask prop with focus
# mask-prop-with-class-focus

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230512-221332" \
#     --is_custom_class 1 \
#     --prefix "focus"

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230513-014610" \
#     --is_custom_class 1 \
#     --prefix "focus"

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230513-023320" \
#     --is_custom_class 1 \
#     --prefix "focus"

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230513-111941" \
#     --is_custom_class 1 \
#     --prefix "focus"

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230513-131604" \
#     --is_custom_class 1 \
#     --prefix "focus"

# python bash_scripts/run_submission_bidirection_mask_aug.py \
#     --model_dir "runs/mask-prop-with-class-focus/mp-focus-230513-151234" \
#     --is_custom_class 1 \
#     --prefix "focus"

# Raw

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-propagation-raw/mask-prop-230508-222109" \
    --is_custom_class 1 \
    --prefix "bi_raw"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-propagation-raw/mask-prop-230509-005503" \
    --is_custom_class 1 \
    --prefix "bi_raw"

python bash_scripts/run_submission_bidirection_mask_aug.py \
    --model_dir "runs/mask-propagation-raw/mask-prop-230511-153918" \
    --is_custom_class 1 \
    --prefix "bi_raw"
