#!/bin/bash

# Raw intitialized, try to overfit
python scripts/experiments/simple_mask_propagate/train_crop_prev_mask.py with \
   learning_rate=6e-4 \
   n_epochs=300 \
   evaluate_epoch=10 \
   save_epoch=10

# First mask-prop trained with lr=6e-4
python scripts/experiments/simple_mask_propagate/train_crop_prev_mask.py with \
    learning_rate=6e-4 \
    custom_model_path="runs/mask-prop-230509-005503/model-100.pt" \
    n_epochs=300 \
    evaluate_epoch=10 \
    save_epoch=10

# Same as above, but train from a one-point 
python scripts/experiments/simple_mask_propagate/train_crop_prev_mask.py with \
    learning_rate=6e-4 \
    custom_model_path="runs/mask-prop-230511-153918/model-100.pt"\
    n_epochs=300 \
    evaluate_epoch=10 \
    save_epoch=10
