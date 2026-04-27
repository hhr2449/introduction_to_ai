# =================== 用来放公共的实验参数和相关配置 ===================

import torch

# ===================== 数据路径 =====================
TRAIN_DATA_PATH = 'Dataset/train.txt'
TEST_DATA_PATH = 'Dataset/test.txt'
VALID_DATA_PATH = 'Dataset/validation.txt'
WORD2VEC_PATH = 'Dataset/wiki_word2vec_50.bin'

# ===================== 基本参数 ======================
SEED = 42
# 句子长度，如果句子长度超过该值，则截断，否则填充
MAX_SENTENCE_LEN = 120
# 词向量的维度
EMBEDDING_DIM = 50
# 类数
NUM_CLASSES = 2

# ===================== 训练参数 ======================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
INIT_METHOD = "default"

# 设备
# 检查如果有可用的GPU，则使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 早停 =========
USE_EARLY_STOPPING = True
PATIENCE = 3
MIN_DELTA = 0.0

# ========= 输出目录 =========
CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"

PAD_TOKEN = "<PAD>"
