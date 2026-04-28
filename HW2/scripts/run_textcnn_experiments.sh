#!/bin/bash
set -e

# =========================
# 1) learning rate
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_lr1e-4 \
  --learning_rate 1e-4 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_lr5e-4 \
  --learning_rate 5e-4 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_lr1e-3 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

# =========================
# 2) batch size
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_bs32 \
  --learning_rate 1e-3 --batch_size 32 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_bs64 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_bs128 \
  --learning_rate 1e-3 --batch_size 128 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

# =========================
# 3) filter sizes
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_fs345 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_fs35 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 5

python3 -m train.train --model textcnn --exp_name textcnn_fs357 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 5 7

# =========================
# 4) num filters
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_nf50 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 50 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_nf100 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_nf200 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 200 --filter_sizes 3 4 5

# =========================
# 5) dropout
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_do03 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.3 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_do05 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_do07 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.7 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

# =========================
# 6) sentence length
# =========================
python3 -m train.train --model textcnn --exp_name textcnn_len40 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 40 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_len60 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 60 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5

python3 -m train.train --model textcnn --exp_name textcnn_len100 \
  --learning_rate 1e-3 --batch_size 64 --dropout 0.5 --weight_decay 1e-4 \
  --max_len 100 --epochs 15 --patience 3 --num_filters 100 --filter_sizes 3 4 5