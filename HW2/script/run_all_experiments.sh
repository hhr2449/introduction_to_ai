#!/bin/bash
set -e

bash scripts/run_baselines.sh
bash scripts/run_mlp_experiments.sh
bash scripts/run_textcnn_experiments.sh
bash scripts/run_rnn_experiments.sh
bash scripts/run_init_experiments.sh