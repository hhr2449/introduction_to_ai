import torch
import torch.nn as nn
import torch.nn.functional as F

from config.cnn_config import (
    EMBEDDING_DIM,
    NUM_CLASSES,
    DROPOUT,
    NUM_FILTERS,
    FILTER_SIZES,
)

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        embedding_matrix=None,
        freeze_embedding=False,
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        dropout=DROPOUT,
    ):
        # 与mlp类似，都是做一些初始化
        super().__init__()
        self.pad_idx = pad_idx
        # 如果传入的是 numpy.ndarray，先转成 torch.FloatTensor
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

        # 开始定义cnn的结构
        # 输入通道数：1； 输出通道数：num_filters
        # 卷积核形状：(filter_size, embedding_dim)
        model_list = []
        # 遍历filter_sizes,创建不同尺寸的卷积核
        for filter_size in filter_sizes:
            model_list.append(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(filter_size, embedding_dim),
                )
            )
        self.convs = nn.ModuleList(model_list)

        # 舍弃层
        self.dropout = nn.Dropout(dropout)

        # 全连接层
        # 每一个输出通道的输出向量会进行最大池化，池化的结果再过全连接，所以输入通道数是num_filters * len(filter_sizes)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids):
        # 词嵌入
        embedded = self.embedding(input_ids)
        # 增加一个维度，变成 (batch_size, 1, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)
        # 卷积的结果
        convs_out = []

        # 对每个卷积核进行一次卷积
        for conv in self.convs:
            # 卷积
            conv_out = conv(embedded)
            # 此时经过卷积后最后一个维度大小为1，将其去掉
            conv_out = conv_out.squeeze(3)
            # 激活函数
            conv_out = F.relu(conv_out)
            # 最大池化
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            # 池化后最后一维大小为1，将其去掉
            conv_out = conv_out.squeeze(2)
            convs_out.append(conv_out)
        
        # 拼接所有卷积核的输出
        convs_out = torch.cat(convs_out, dim=1)
        # 做舍弃
        convs_out = self.dropout(convs_out)
        # 经过全连接
        logits = self.fc(convs_out)
        return logits
                
        
