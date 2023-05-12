#!/bin/bash

python scripts/experiments/simple_mask_propagate/train.py with \
    learning_rate=6e-4 \
    n_epochs = 200

python scripts/experiments/simple_mask_propagate/train.py with \
    learning_rate=6e-4 \
    custom_model_path=...

python scripts/experiments/simple_mask_propagate/train.py with \
    learning_rate=6e-4 \
    custom_model_path=...
    
