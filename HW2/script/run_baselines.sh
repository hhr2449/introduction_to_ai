#!/bin/bash
set -e

python -m train.train --model mlp --exp_name mlp_base

python -m train.train --model textcnn --exp_name textcnn_base \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --max_len 60 \
  --epochs 15 \
  --patience 3 \
  --num_filters 100 \
  --filter_sizes 3 4 5 \
  --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_base \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --max_len 60 \
  --epochs 15 \
  --patience 3 \
  --hidden_size 128 \
  --num_layers 1 \
  --bidirectional true \
  --init_method default