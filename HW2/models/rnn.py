import torch
import torch.nn as nn

from config.rnn_config import (
    EMBEDDING_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
    DROPOUT,
    BIDIRECTIONAL,
)


class RNN_LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        embedding_matrix=None,
        freeze_embedding=False,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if embedding_matrix is not None and not isinstance(embedding_matrix, torch.Tensor):
            embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
            _weight=embedding_matrix,
        )
        # 是否冻结词嵌入参数
        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        # 开始定义rnn结构
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 舍弃层
        self.dropout = nn.Dropout(dropout)

        # 如果是双向输出，则输出向量要进行拼接，维度乘2
        if bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        # 经过LSTM
        _, (hidden, _) = self.rnn(embedded)

        # 将双向输出进行拼接
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # 舍弃
        hidden = self.dropout(hidden)
        # 线性层
        logits = self.fc(hidden)

        return logits

