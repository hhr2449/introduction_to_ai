#!/bin/bash
set -e

# =========================
# 1) hidden size
# =========================
python -m train.train --model rnn_lstm --exp_name rnn_h64 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 64 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_h128 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_h256 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 256 --num_layers 1 --bidirectional true --init_method default

# =========================
# 2) num_layers
# =========================
python -m train.train --model rnn_lstm --exp_name rnn_l1 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_l2 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 2 --bidirectional true --init_method default

# =========================
# 3) bidirectional
# =========================
python -m train.train --model rnn_lstm --exp_name rnn_bi_true \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_bi_false \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional false --init_method default

# =========================
# 4) sentence length
# =========================
python -m train.train --model rnn_lstm --exp_name rnn_len40 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 40 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_len60 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default

python -m train.train --model rnn_lstm --exp_name rnn_len100 \
  --batch_size 32 --learning_rate 5e-4 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 100 --epochs 15 --patience 3 \
  --hidden_size 128 --num_layers 1 --bidirectional true --init_method default