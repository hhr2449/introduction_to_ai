from .base_config import *

MODEL_NAME = "rnn"

# ========= RNN / LSTM 结构参数 =========
HIDDEN_SIZE = 128
NUM_LAYERS = 1
BIDIRECTIONAL = True

# ========= RNN / LSTM最优默认参数 =========
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
MAX_SENTENCE_LEN = 60
EPOCHS = 15
INIT_METHOD = "xavier"

# ========= 早停 =========
USE_EARLY_STOPPING = True
PATIENCE = 3

# ========= 保存路径 =========
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/rnn_best.pt"
