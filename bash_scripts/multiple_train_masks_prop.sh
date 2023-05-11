#!/bin/bash
python scripts/experiments/simple_mask_propagate/train.py with \
    learning_rate=6e-4 \
    custom_model_path="runs/sam-one-point-230501-152140/model-40.pt"

