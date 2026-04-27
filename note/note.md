# 实验要求

## 简介

给定一个句子，要求对句子进行二分类，类别包含正向和负向两种

## 数据

训练、验证、测试集分别为`train.txt、validation.txt、test.txt`

格式：每行代表一个样本，行首为标签，0代表负类、1代表正类（数据中的正类其实是负向情感、负类是正向情感，和主流定义相反，但是不影响训练和测试）

后面跟着的是一个已经完成了分词的句子



词向量：`wiki_word2vec_50.bin`已经训练好了的50维词向量文件，可以使用gensim库进行加载

## 要求

实现CNN,RNN模型

额外实现baseline模型（如MLT）



# pytorch相关知识

## dataset与dataloader

**torch.utils.data.Dataset** 是一个抽象类，允许你从自己的数据源中创建数据集。

需要继承该类并实现以下两个方法：

- `__len__(self)`：返回数据集中的样本数量。
- `__getitem__(self, idx)`：通过索引返回一个样本。

例如：

```python
import torch
from torch.utils.data import Dataset

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y

# 示例数据
X_data = [[1, 2], [3, 4], [5, 6], [7, 8]]  # 输入特征
Y_data = [1, 0, 1, 0]  # 目标标签

# 创建数据集实例
dataset = MyDataset(X_data, Y_data)
```

dataloader用于从dataset种按批次加载数据

再开始训练前，需要将训练数据表示成dataset，然后训练的时候就可以使用dataloader按批次进行加载



## pytorch中构建模型的基本框架

```python
import torch
import torch.nn as nn

# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 2. 实例化模型
model = MyModel()

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. 假设一批输入数据
x = torch.randn(4, 10)                  # 4个样本，每个样本10维
labels = torch.tensor([0, 1, 1, 0])     # 4个标签

# 6. 训练一步
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

print(loss.item())
```

### 定义模型

#### 模型类

需要创建模型类`MyModel`，这个类继承自`nn.Module`

实现两个方法`__init__()`和`forward()`

`__init__()`中定义了模型有哪些层

`forword()`定义了前向传播的流程，也就是输入一个数据，需要经过哪些层级，最后返回结果

### 定义损失函数

`criterion = nn.CrossEntropyLoss()`使用交叉熵损失函数

### 定义优化器

`optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`

`model.parameters()`模型中所有可以进行学习的参数，只要模型类继承了nn.Module就可以用

`lr`学习率

`Adam`更新参数的方法

优化器负责真的去改动参数

### 训练循环

```python
for input_ids, labels in train_loader:
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

1. 从训练数据集中取出一份数据
2. `optimizer.zero_grad()`将梯度清空
3. `logits = model(input_ids)`将数据输入模型，得到输出
4. `loss = criterion(logits, labels)`获取损失
5. `loss.backward()`进行反向传播
6. `optimizer.step()`根据反向传播得到的梯度进行实际更新



## torch.nn

### nn.Module

**所有神经网络的基类**。继承它之后，PyTorch 才会自动追踪你定义的层和参数，并允许你使用 `model(x)` 进行前向传播。

### nn.Embedding

词嵌入层，原本的一个句子被表示成一个id列表，每个元素都对应一个词在词表中的id

词嵌入层就是将id替换成词向量

若输入形状为 $(N, W)$，输出形状为 $(N, W, \text{embedding\_dim})$

#### 初始化

`self.embedding = nn.Embedding(...)`

必填参数：

1. num_embeddings：词表大小
2. embedding_dim：词向量的维度

选填参数：

padding_idx：指定哪个id是填充符，该id对应的词向量固定为0并且训练过程中无梯度

#### 核心属性与方法

`.weight`：储存词向量的权重矩阵

形状为`(num_embeddings, embedding_dim)`，每行对应一个词向量

注意其为可训练参数，过程中会进行更新，如果不想更新，可以指定

`embedding.weight.requires_grad = False `



`from_pretrained(cls, embeddings, freeze=True, padding_idx=None, ...)`

**`embeddings`**：一个 `FloatTensor`，包含预训练好的权重

**`freeze`**：默认为 `True`。这意味着加载后词向量被冻结，不参与训练。如果你希望模型在你的特定任务上微调（Fine-tune）这些词向量，请设为 `False`

可以使用预训练好的权重

### nn.Linear

nn.Linear(输入维度，输出维度)

全连接层，指定输入维度和输出维度

### nn.Dropout

随机失活层，会自动切换行为

当目前处于训练阶段时，经过该层会按照指定的比例p将一些神经元的输出置为0；处于非训练阶段时什么都不做

### torch.nn.functional

用于调用非线性函数



# 前置工作

## 配置环境

创建一个conda环境

依赖文件如下：

```
torch
numpy
gensim
scikit-learn
tqdm
matplotlib
pandas
```

使用pip进行安装

## config.py

作用：存放实验参数和相关配置

```python
# =================== 用来放实验参数和相关配置 ===================

import torch

# ===================== 数据路径 =====================
TRAIN_DATA_PATH = 'Dataset/train.txt'
TEST_DATA_PATH = 'Dataset/test.txt'
VALID_DATA_PATH = 'Dataset/valid.txt'
WORD2VEC_PATH = 'Dataset/wiki_word2vec_50.bin'

# ===================== 基本参数 ======================

# 句子长度，如果句子长度超过该值，则截断，否则填充
MAX_SENTENCE_LEN = 100
# 词向量的维度
EMBEDDING_DIM = 50
# 类数
NUM_CLASSES = 2

# ===================== 训练参数 ======================
# 一次梯度下降使用的样本大小维batch_size
BATCH_SIZE = 128
# 训练的轮次，也就是要重复训练epochs次
EPOCHS = 10
# 学习率
LEARNING_RATE = 0.01
# 丢弃率
DROPOUT = 0.5

# ==================== MLP 参数 =======================
MLP_HIDDEN_SIZE = 128

# 设备
# 检查如果有可用的GPU，则使用GPU，否则使用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出路径
CHECKPOINT_DIR = "checkpoints"
RESULT_DIR = "results"

```

## dataset.py

用于数据预处理，将原始训练数据转换成可以直接用于训练的数据

1. 读取文本文件，将文件读成python列表

2. 训练数据中的句子是一个一个的词，需要建立词表，将句子转换成`词表id列表`的形式

   比如建立词表：

   ```python
   word2idx = {
       "<PAD>": 0,
       "<UNK>": 1,
       "这": 2,
       "部": 3,
       "电影": 4,
       "很": 5,
       "好看": 6
   }
   ```

   句子：`["这", "部", "电影", "很", "好看"]`可以转换为`[2, 3, 4, 5, 6]`

3. 整理句子长度，如果长度不够长，就将其补到指定的长度；如果过长就截断

4. 将样本组织成batch供训练时使用



关键是实现Dataset类，这个类里面要提供`__len__`和`__getitem__`方法

然后实例化Dataset，就可以构建Dataloader了

```python
import os
from config import *
import torch
from torch.utils.data import Dataset, DataLoader


# 用于读取数据集
# 输入数据集的路径
# 返回一个二元组的列表，其中的每个二元组都表示一条训练数据，具体而言是：(label, sentence)
# sentence是一个列表，列表中的每个元素表示一个单词
def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 先去除首末空格
            line = line.strip()
            # 然后按照空格分隔
            words = line.split()
            # 构建一个二元组
            data_line = (int(words[0]), words[1:])
            # 添加到列表中
            data.append(data_line)
    
    # 返回
    return data


# 构建词表
# 输入数据data，data定义参照read_data_from_file()
# mini_freq表示最小词频，小于该词频的词将被忽略
#返回一个字典word2id,里面建立的词和id的对应关系
def build_vocab(data, mini_freq=1):
    word2id = {}
    # 初始化: 0 为<pad>表示填充占位符
    # 1 为<unk>表示未知字符
    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1
    # 接下来遍历数据集，开始统计词频
    word_freq = {}
    # 遍历句子
    for label, sentence in data:
        # 遍历句子中的单词
        for word in sentence:
            # 词频加1
            word_freq[word] = word_freq.get(word, 0) + 1
    # 对于词频表中的词，只要满足最小词频要求，则添加到词表中
    for word, freq in word_freq.items():
        if freq >= mini_freq:
            word2id[word] = len(word2id)
    # 同时构建id到词的映射
    id2word = {v: k for k, v in word2id.items()}

    return word2id, id2word

# 将数据转换为id表示
# 输入数据data，data定义参照read_data_from_file()
# word2id，定义参照build_vocab()
# 输出一个二元组的列表，其中的每个二元组都表示一条训练数据，具体而言是：(label, sentence_ids)
# sentence_ids是一个列表，列表中的每个元素表示一个单词的id，长度为MAX_SENTENCE_LEN
def convert_data_to_id(data, word2id):
    data_ids = []
    max_len = MAX_SENTENCE_LEN
    pad_id = word2id['<PAD>']
    unk_id = word2id['<UNK>']

    for label, sentence in data:
        # 先创建一个全都是pad的列表
        sentence_ids = [pad_id] * max_len
        # 然后对于sentence中的单词，将其转换为id，并替换列表中的元素
        limit = min(len(sentence), max_len)
        for i in range(limit):
            sentence_ids[i] = word2id.get(sentence[i], unk_id)
        data_ids.append((label, sentence_ids))

    return data_ids

# 数据集类,继承torch.utils.data.Dataset
class SentimentDataset(Dataset):

    def __init__(self, data_ids, word2id, max_len=MAX_SENTENCE_LEN):
        self.data_ids = data_ids
        self.word2id = word2id
        self.max_len = max_len
    
    # 返回数据集长度
    def __len__(self):
        return len(self.data_ids)

    # 获取索引为idx的数据
    # 返回sentence_ids和label的tensor
    def __getitem__(self, index):
        label, sentence_ids = self.data_ids[index]
        return torch.tensor(sentence_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
def get_data_loaders():
    # 读取数据
    train_data = read_data_from_file(TRAIN_DATA_PATH)
    valid_data = read_data_from_file(VALID_DATA_PATH)
    test_data = read_data_from_file(TEST_DATA_PATH)

    # 构建词表
    # 只使用训练数据中的词
    word2id, id2word = build_vocab(train_data)
    # 将数据转换为id表示
    train_data_ids = convert_data_to_id(train_data, word2id)
    valid_data_ids = convert_data_to_id(valid_data, word2id)
    test_data_ids = convert_data_to_id(test_data, word2id)

    # 实例化Dataset
    train_dataset = SentimentDataset(train_data_ids, word2id)
    valid_dataset = SentimentDataset(valid_data_ids, word2id)
    test_dataset = SentimentDataset(test_data_ids, word2id)

    # 实例化DataLoader
    # 需要输入Dataset和batch_size，shuffle表示是否打乱数据,这里只选择训练数据打乱
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 返回DataLoader和词表
    return train_data_loader, valid_data_loader, test_data_loader, word2id, id2word
```

# 构建模型

## pytorch中构建模型的基本框架

```python
import torch
import torch.nn as nn

# 1. 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 2. 实例化模型
model = MyModel()

# 3. 定义损失函数
criterion = nn.CrossEntropyLoss()

# 4. 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. 假设一批输入数据
x = torch.randn(4, 10)                  # 4个样本，每个样本10维
labels = torch.tensor([0, 1, 1, 0])     # 4个标签

# 6. 训练一步
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()

print(loss.item())
```

### 定义模型

#### 模型类

需要创建模型类`MyModel`，这个类继承自`nn.Module`

实现两个方法`__init__()`和`forward()`

`__init__()`中定义了模型有哪些层

`forword()`定义了前向传播的流程，也就是输入一个数据，需要经过哪些层级，最后返回结果

`forward(input_ids)`的输入是一个`[batch_size, max_len]`的tensor，也就是一次可以输入一个batch的训练数据

### 定义损失函数

`criterion = nn.CrossEntropyLoss()`使用交叉熵损失函数

### 定义优化器

`optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`

`model.parameters()`模型中所有可以进行学习的参数，只要模型类继承了nn.Module就可以用

`lr`学习率

`Adam`更新参数的方法

优化器负责真的去改动参数

### 训练循环

```python
for input_ids, labels in train_loader:
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

1. 从训练数据集中取出一份数据
2. `optimizer.zero_grad()`将梯度清空
3. `logits = model(input_ids)`将数据输入模型，得到输出
4. `loss = criterion(logits, labels)`获取损失
5. `loss.backward()`进行反向传播
6. `optimizer.step()`根据反向传播得到的梯度进行实际更新

## MLP

### 基本模型类定义

```python
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
```

### 训练过程

#### 做一轮训练

需要传入：`model`、`data_loader`、`criterion(损失函数)`、`optimizer(优化器)`、`device`

1. 切换到训练模式

   `model.train()`因为有些层在训练模式和验证模式下的行为不一样

2. 开始迭代data_loader

   - 清空梯度
   - 前向传播
   - 计算损失
   - 反向传播
   - 优化器更新参数



#### 做验证

不计算梯度和更新梯度，只是进行前向传播和计算损失正确率

用于在验证集上验证和测试集shang