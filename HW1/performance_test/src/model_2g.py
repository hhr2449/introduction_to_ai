'''
model_2g.py
用于2元模型
ProbCalculator2G用于计算概率，运行viterbi算法
ParserPinyin2G用于读取拼音并且进行解码
'''

import math

class ProbCalculator2G:
    # corpus是通过语料训练出来的词频等
    # lam和alpha是参数
    def __init__(self, corpus, lam=0.9, alpha=0.1):
        self.corpus = corpus
        self.lam = lam
        self.alpha = alpha

        # 缓存，虽然对于测试文件好像无用
        self.uniword_cache = {}

    # 计算一元概率
    def get_unigram_prob(self, word):
        # 如果缓存中存在，则直接返回
        if word in self.uniword_cache:
            return self.uniword_cache[word]
        # 如果没命中，进行计算
        count = self.corpus.uniword_table.get(word, 0)
        total = self.corpus.uni_count
        num_of_types = self.corpus.num_of_types
        # 这里进行了平滑处理，如果字没有出现过的话频数即为0
        # 所以零分子加上1，分母加上总频数，防止出0
        self.uniword_cache[word] = (count + self.alpha) / (total + self.alpha * num_of_types)
        return self.uniword_cache[word]
    
    # 计算二元概率
    def get_bigram_prob(self, word1, word2):
        # 内存不够，不做缓存了

        # 1. 计算极大似然估计 P_ML(w2 | w1)
        # count(w1, w2) / count(w1)
        c12 = self.corpus.bigram_table.get((word1, word2), 0)
        c1 = self.corpus.uniword_table.get(word1, 0)
        p_ml = c12 / c1 if c1 > 0 else 0
        
        # 2. 获取 P(w2) 的概率 
        p_w2 = self.get_unigram_prob(word2)
        
        # 3. 线性插值公式
        prob = self.lam * p_ml + (1 - self.lam) * p_w2
        
        return prob
    
    # 获取概率的负对数，即使用的代价
    def get_unigram(self, word):
        return -math.log(self.get_unigram_prob(word))
    
    def get_bigram(self, word1, word2):
        return -math.log(self.get_bigram_prob(word1, word2))


class ViterbiNode:
    # 优化内存
    __slots__ = ['word', 'weight', 'prev_node']
    def __init__(self, word, weight, prev_node):
        self.word = word            # 当前汉字
        self.weight = weight        # 从起点到这里的最小负对数概率之和
        self.prev_node = prev_node  # 最优前驱


# 维特比算法
# 输入：拼音列表，语料，概率计算器
def viterbi(pinyinList, corpus, calculator):
    # 建立一个二维的维特比数组
    # 每一列都是一个拼音对应的字符候选列表
    dp_table = []

    # 填充第一列，起始概率就是单字概率
    first_candidates = corpus.pinyin_table.get(pinyinList[0], [])
    first_column = []
    for candidate in first_candidates:
        first_column.append(ViterbiNode(candidate, calculator.get_unigram(candidate), None))
    dp_table.append(first_column)

    # 往后递推
    for i in range(1, len(pinyinList)):
        candidates = corpus.pinyin_table.get(pinyinList[i], [])
        column = []
        prev_column = dp_table[i - 1]
        # 遍历该列所有字符
        for candidate in candidates:
            # 初始化最优权重和最优前驱
            best_weight = float('inf')
            best_prev = None
            # 遍历上一列的所有节点并且进行松弛操作
            for j in range(len(prev_column)):
                prev_node = prev_column[j]
                # 计算权重
                weight = prev_node.weight + calculator.get_bigram(prev_node.word, candidate)
                # 松弛
                if weight < best_weight:
                    best_weight = weight
                    best_prev = prev_node
            # 此时已经获取了最优权重和最优前驱
            column.append(ViterbiNode(candidate, best_weight, best_prev))
        dp_table.append(column)

    # 此时已经填完了表，只需找最后一个列的最优解，回溯
    best_node = None
    best_weight = float('inf')
    for node in dp_table[-1]:
        if node.weight < best_weight:
            best_weight = node.weight
            best_node = node
    # 回溯
    res = []
    while best_node:
        res.append(best_node.word)
        best_node = best_node.prev_node
    return res[::-1]


# 解析一个拼音串
class ParserPinyin2G:
    def __init__(self, corpus, lam=0.89, alpha=0.1):
        self.corpus = corpus
        self.calculator = ProbCalculator2G(corpus, lam=lam, alpha=alpha)
    # 解析拼音列表
    def parser_pinyin_list(self, pinyin_list):
        res = viterbi(pinyin_list, self.corpus, self.calculator)
        return "".join(res)
    
    # 解析拼音串
    def parser_pinyin_string(self, pinyin_string):
        pinyin_string = pinyin_string.strip()
        if not pinyin_string:
            return ""
        
        pinyin_list = pinyin_string.split()
        return self.parser_pinyin_list(pinyin_list)