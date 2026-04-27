#!/bin/bash
set -e

python -m train.train --model mlp --exp_name mlp_bs32 --batch_size 32
python -m train.train --model mlp --exp_name mlp_bs64 --batch_size 64
python -m train.train --model mlp --exp_name mlp_bs128 --batch_size 128

python -m train.train --model mlp --exp_name mlp_lr1e-4 --learning_rate 1e-4
python -m train.train --model mlp --exp_name mlp_lr5e-4 --learning_rate 5e-4
python -m train.train --model mlp --exp_name mlp_lr1e-3 --learning_rate 1e-3

python -m train.train --model mlp --exp_name mlp_do03 --dropout 0.3
python -m train.train --model mlp --exp_name mlp_do05 --dropout 0.5
python -m train.train --model mlp --exp_name mlp_do07 --dropout 0.7

python -m train.train --model mlp --exp_name mlp_len40 --max_len 40
python -m train.train --model mlp --exp_name mlp_len60 --max_len 60
python -m train.train --model mlp --exp_name mlp_len100 --max_len 100