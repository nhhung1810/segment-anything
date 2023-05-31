#!/bin/bash

# Raw intitialized, try to overfit
# python scripts/experiments/mask_aug/train.py with \
#    learning_rate=6e-4 \
#    n_epochs=300 \
#    evaluate_epoch=10 \
#    save_epoch=10

# First mask-prop trained with lr=6e-5, 
# class-focus init inside the train code
python scripts/experiments/mask_aug/train.py with \
    learning_rate=6e-5 \
    custom_model_path="runs/mask-liver-first-augment/mask-drop-230518-214209/model-160.pt" \
    evaluate_epoch=5 \
    save_epoch=10

# python scripts/experiments/mask_aug/train.py with \
#     learning_rate=6e-6 \
#     custom_model_path="runs/mask-liver-first-augment/mask-drop-230518-214209/model-160.pt" \
#     n_epochs=300 \
#     evaluate_epoch=10 \
#     save_epoch=10

