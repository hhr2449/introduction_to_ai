# cnn参数

from .base_config import *

MODEL_NAME = "cnn"

# ========= CNN结构参数 =========
# 卷积核的数量，这里设定每种卷积核有100个
NUM_FILTERS = 100
# 卷积核的大小
FILTER_SIZES = [3, 4, 5]

# ========= CNN训练参数 =========
LEARNING_RATE = 1e-3
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4

# ========= 早停参数 =========
USE_EARLY_STOPPING = True
PATIENCE = 3

# ========= 模型保存路径 =========
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/cnn_best.pt"