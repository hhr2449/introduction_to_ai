import os

import numpy as np
from config.base_config import (
    BATCH_SIZE,
    MAX_SENTENCE_LEN,
    TRAIN_DATA_PATH,
    VALID_DATA_PATH,
    TEST_DATA_PATH,
    EMBEDDING_DIM,
    WORD2VEC_PATH,
)
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors


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
    
# 加载词向量矩阵
def build_embedding_matrix(word2id, word2vec_path=WORD2VEC_PATH, embedding_dim=EMBEDDING_DIM):
    # 加载预训练的词向量
    # 被储存为字典，key为单词，value为向量
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # 创建词向量矩阵，行数为词表大小，列数为词向量的维度，每一行对应一个词向量，行数就是词的id
    vocab_size = len(word2id)
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.1,
        size=(vocab_size, embedding_dim)
    ).astype(np.float32)
    # pad词向量设置为全0
    pad_idx = word2id['<PAD>']
    embedding_matrix[pad_idx] = np.zeros((embedding_dim,))
    # 遍历词表，进行填充
    for word, i in word2id.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word]

    return embedding_matrix



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

    # 构建词向量矩阵
    embedding_matrix = build_embedding_matrix(word2id)

    # 返回DataLoader和词表
    return train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, embedding_matrix