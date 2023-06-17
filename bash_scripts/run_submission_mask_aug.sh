#!/bin/bash
# "imp-aug-230606-002414"
candidates=("imp-aug-230610-104249")
# candidates=("imp-aug-230605-000452" "imp-aug-230603-215252" "imp-aug-230605-165716")
for t in ${candidates[@]}; do
    echo "runs/$t"
    python bash_scripts/run_submission_mask_aug.py \
        --model_dir "runs/$t" \
        --prefix "thesis"
        # --is_custom_class 1 \
done