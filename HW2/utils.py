import random
import numpy as np
import torch

# 固定所有随机数发生器的种子，保证结果稳定可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)