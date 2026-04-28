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
from collections import Counter
from itertools import chain


# 用于读取数据集
# 输入数据集的路径
# 返回一个二元组的列表，其中的每个二元组都表示一条训练数据，具体而言是：(label, sentence)
# sentence是一个列表，列表中的每个元素表示一个单词
def read_data_from_file(file_path):
    data_init = []
    for i in range(1, 10):
        data_init.append(read_data_from_file(os.path.join(file_path, 'train_' + str(i) + '.txt')))
        data_init.append(read_data_from_file(os.path.join(file_path, 'valid_' + str(i) + '.txt')))
        data_init.append(read_data_from_file(os.path.join(file_path, 'test_' + str(i) + '.txt')))

    


    
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
    # 1. 扁平化提取所有词，并使用 Counter 高效统计词频
    all_words = chain.from_iterable(sentence for _, sentence in data)
    word_freq = Counter(all_words)
    
    # 2. 初始化 word2id 并利用 enumerate 批量分配 ID
    word2id = {'<PAD>': 0, '<UNK>': 1}
    valid_words = (word for word, freq in word_freq.items() if freq >= mini_freq)
    word2id.update({word: i for i, word in enumerate(valid_words, start=2)})
    
    # 3. 字典推导式构建反向映射
    id2word = {v: k for k, v in word2id.items()}
    
    return word2id, id2word

# 将数据转换为id表示
# 输入数据data，data定义参照read_data_from_file()
# word2id，定义参照build_vocab()
# 输出一个二元组的列表，其中的每个二元组都表示一条训练数据，具体而言是：(label, sentence_ids)
# sentence_ids是一个列表，列表中的每个元素表示一个单词的id，长度为MAX_SENTENCE_LEN
def convert_data_to_id(data, word2id, max_len=MAX_SENTENCE_LEN):
    result_list = []
    
   
    pad_val = word2id['<PAD>']
    unk_val = word2id['<UNK>']

    for item in data:
        cur_label = item[0]
        cur_text = item[1]
        
        tmp_ids = []
        word_count = 0  
        
        # 逐个单词处理
        for word in cur_text:
            
            if word_count >= max_len:
                break
                
            
            if word in word2id.keys():
                tmp_ids.append(word2id[word])
            else:
                tmp_ids.append(unk_val)
                
            word_count += 1
            
        
        current_length = len(tmp_ids)
        while current_length < max_len:
            tmp_ids.append(pad_val)
            current_length += 1
            
        result_list.append((cur_label, tmp_ids))

    return result_list

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



def get_data_loaders(batch_size=BATCH_SIZE, max_len=MAX_SENTENCE_LEN):
    # 读取数据
    train_data = read_data_from_file(TRAIN_DATA_PATH)
    valid_data = read_data_from_file(VALID_DATA_PATH)
    test_data = read_data_from_file(TEST_DATA_PATH)

    # 构建词表
    # 只使用训练数据中的词
    word2id, id2word = build_vocab(train_data)
    # 将数据转换为id表示
    train_data_ids = convert_data_to_id(train_data, word2id, max_len=max_len)
    valid_data_ids = convert_data_to_id(valid_data, word2id, max_len=max_len)
    test_data_ids = convert_data_to_id(test_data, word2id, max_len=max_len)

    # 实例化Dataset
    train_dataset = SentimentDataset(train_data_ids, word2id, max_len=max_len)
    valid_dataset = SentimentDataset(valid_data_ids, word2id, max_len=max_len)
    test_dataset = SentimentDataset(test_data_ids, word2id, max_len=max_len)

    # 实例化DataLoader
    # 需要输入Dataset和batch_size，shuffle表示是否打乱数据,这里只选择训练数据打乱
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 构建词向量矩阵
    embedding_matrix = build_embedding_matrix(word2id)

    # 返回DataLoader和词表
    return train_data_loader, valid_data_loader, test_data_loader, word2id, id2word, embedding_matrix
