#!/bin/bash
python scripts/experiments/improve_mask_prop/train.py with 'class_selected=[1, 9]' \
    'train_n_previous_frame=3'

# python scripts/experiments/improve_mask_prop/train.py with 'class_selected=[1, 9]'\
#     custom_model_path="runs/imp-230601-213326/model-15.pt"\
#     'train_n_previous_frame=3'


# python scripts/experiments/improve_mask_prop/train.py with 'class_selected=[1, 9, 6, 2]'\
#     'train_n_previous_frame=3'

