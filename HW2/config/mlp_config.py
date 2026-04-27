# mlp的参数设置
from .base_config import *

MODEL_NAME = "mlp"

# MLP结构参数
MLP_HIDDEN_SIZE = 128

# MLP训练参数(默认采用与base_config一致的，可以自行调整)
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.5
# 正则化强度
WEIGHT_DECAY = 1e-4  

CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/mlp_best.pt"