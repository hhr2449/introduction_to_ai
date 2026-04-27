from .base_config import *

MODEL_NAME = "rnn"

# ========= RNN / LSTM 结构参数 =========
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True

# ========= 训练参数 =========
LEARNING_RATE = 5e-4
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4

# ========= 早停 =========
USE_EARLY_STOPPING = True
PATIENCE = 3

# ========= 保存路径 =========
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/rnn_best.pt"