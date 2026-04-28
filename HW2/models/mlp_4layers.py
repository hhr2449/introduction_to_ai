# 实现mlp模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.mlp_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    DROPOUT,
)

# 创建MLP模型类，要继承nn.Module类
class MLP(nn.Module):
    # 定义模型结构
    def __init__(
        self,
        vocab_size,
        pad_idx,
        embedding_matrix=None,
        freeze_embedding=False,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        # 如果传入的是 numpy.ndarray，先转成 torch.FloatTensor
        if embedding_matrix is not None and not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

        # 词嵌入层
        # 将输入的id列表转换为词向量列表
        # embedding_matrix为预训练的词向量矩阵，如果为None，则使用随机初始化的词向量矩阵
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
            _weight=embedding_matrix,
        )
        # 是否冻结词嵌入参数
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        
        # 四层MLP隐藏层: [512, 256, 128, 64]
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        # dropout层
        self.dropout = nn.Dropout(dropout) 

    # 前向传播
    # 输入一个句子的id列表，返回预测值
    def forward(self, input_ids):
        # 词嵌入
        embedded = self.embedding(input_ids)

        # 对词嵌入后的句子进行平均池化（词向量取平均值）
        mask = (input_ids != self.pad_idx).float()
        mask = mask.unsqueeze(-1)

        embedded = embedded * mask

        lengths = mask.sum(dim=1)
        lengths = lengths.clamp(min=1.0)

        sentence_embedding = embedded.sum(dim=1) / lengths

        x = self.fc1(sentence_embedding)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        # 不再做softmax了，直接使用nn.CrossEntropyLoss()损失函数，它会先做softmax再计算损失
        return x
