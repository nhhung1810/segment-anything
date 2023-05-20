#!/bin/bash

# Raw intitialized, try to overfit
python scripts/experiments/mask_aug/train.py with \
   learning_rate=6e-4 \
   n_epochs=300 \
   evaluate_epoch=10 \
   save_epoch=10

# First mask-prop trained with lr=6e-5, 
# class-focus init inside the train code
python scripts/experiments/mask_aug/train.py with \
    learning_rate=6e-5 \
    custom_model_path="runs/mask-aug-230520-140411/model-300.pt" \
    n_epochs=300 \
    evaluate_epoch=10 \
    save_epoch=10


