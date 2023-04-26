#!/bin/bash


python scripts/train/train.py with learning_rate=6e-4 
python scripts/train/train.py with learning_rate=6e-6
python scripts/train/train.py with learning_rate=6e-5 learning_rate_decay_steps=5 learning_rate_decay_rate = 0.98