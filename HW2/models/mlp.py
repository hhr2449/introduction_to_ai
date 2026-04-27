# 实现mlp模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.mlp_config import (
    EMBEDDING_DIM,
    MLP_HIDDEN_SIZE,
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
        hidden_dim=MLP_HIDDEN_SIZE,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.pad_idx = pad_idx

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
        
        # 第一个全连接（输入层到隐藏层）
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        # 第二个全连接（隐藏层到输出层）
        self.fc2 = nn.Linear(hidden_dim, num_classes)
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

        # 过全连接
        x = self.fc1(sentence_embedding)
        # 激活函数
        x = F.relu(x)
        # 随机丢弃
        x = self.dropout(x)
        # 经过第二个全连接
        x = self.fc2(x)
        # 不再做softmax了，直接使用nn.CrossEntropyLoss()损失函数，它会先做softmax再计算损失

