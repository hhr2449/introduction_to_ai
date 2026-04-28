#!/bin/bash
set -e

# =========================
# 1) MLP vs 4-layer MLP
# =========================
python3 -m train.train --model mlp --exp_name mlp_base
python3 -m train.train --model mlp_4layers --exp_name mlp_4layers_base

# =========================
# 2) Early stopping vs fixed epochs
# =========================

# MLP
python3 -m train.train --model mlp --exp_name mlp_es_on \
  --use_early_stopping true --epochs 15 --patience 3

python3 -m train.train --model mlp --exp_name mlp_es_off \
  --use_early_stopping false --epochs 15

# TextCNN
python3 -m train.train --model textcnn --exp_name textcnn_es_on \
  --use_early_stopping true --epochs 15 --patience 3

python3 -m train.train --model textcnn --exp_name textcnn_es_off \
  --use_early_stopping false --epochs 15

# RNN-LSTM
python3 -m train.train --model rnn_lstm --exp_name rnn_es_on \
  --use_early_stopping true --epochs 15 --patience 3

python3 -m train.train --model rnn_lstm --exp_name rnn_es_off \
  --use_early_stopping false --epochs 15

# =========================
# 3) Weight decay experiments
# =========================

# TextCNN
python3 -m train.train --model textcnn --exp_name textcnn_wd0 --weight_decay 0
python3 -m train.train --model textcnn --exp_name textcnn_wd1e-4 --weight_decay 1e-4
python3 -m train.train --model textcnn --exp_name textcnn_wd1e-3 --weight_decay 1e-3
