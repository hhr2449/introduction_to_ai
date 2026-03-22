# 1.语料的读取与储存：
# oj提供了word2pinyin.txt，1_word.txt,2_word.txt分别为字音表、单字符频率、双字符频率
# 考虑使用一个类来封装这些语料信息：PinyinCorpus
# 一个字典用于字音储存，一个拼音字符串对应一个字符列表
# 使用json解析字符频率表，这里解析的结果是：大括号被解析为字典，[]被解析为列表

# 2.运行算法
# 过程中创建两个字典用于缓存字和字组的概率负对数
# 一元表 (Unigram)： Dict[str, float]，例如 {'清': -4.5, '华': -3.2}
# 二元表 (Bigram)： Dict[Tuple[str, str], float]，例如 {('清', '华'): -2.1, ('大', '学'): -1.5}
# 遇到一个字或是字组的时候，先去缓存中寻找，如果为命中，则计算并缓存
# 可以创建一个类，封装缓存信息+计算函数，对外提供get接口，接口函数去查缓存或是调用计算函数，
# 将之前的语料信息类的实例放在这个类中作为属性

import sys
import json
import math

class PinyinCorpus:
    # 构造函数，需要提供文件路径
    def __init__(self, word2pinyin_file, word_file, bigram_file):
        # 拼音表，字的种类数
        self.pinyintable, self.num_of_types = self.load_pinyin(word2pinyin_file)

        # 直接json解析后的表，临时储存，用于后面的统计和扁平表的转换
        raw_uniwordtable = self.load_json(word_file)
        raw_bigramtable = self.load_json(bigram_file)
        # 经过扁平化的两个表
        # 单字表：字-->频数的字典, 单字总频数
        self.flat_uniwordtable, self.uni_count = self.init_flat_uniwordtable(raw_uniwordtable) 
        # 双字表：字组-->频数的字典，双字总频数
        self.flat_bigramtable, self.bi_count = self.init_flat_bigramtable(raw_bigramtable)
        # 因为要统计总频数，顺便就将单字表和双字表进行初始化，这样计算概率的时候可以节约一点时间
        self.num_of_types = len(self.flat_uniwordtable)


    def init_flat_uniwordtable(self, raw_uniwordtable):
        flat_uniwordtable = {}
        uni_count = 0
        # 遍历raw_uniwordtable，统计频率并且转换为扁平化的表
        # {
        #     <拼音>:{
        #         "words":<列表，代表该读音对应的汉字>,
        #         "counts":<列表，和上方列表一一对应，代表该汉字出现的次数>
        #     },
        #     ......
        # }
        # 对应嵌套字典，外层通过拼音索引一个包含两个列表的字典
        while raw_uniwordtable:
            # 使用popitem()方法，弹出字典中的一个元素，同时将其销毁
            _, data = raw_uniwordtable.popitem()
            words = data.get("words", [])
            counts = data.get("counts", [])
            for i in range(len(words)):
                if words[i] not in flat_uniwordtable:
                    flat_uniwordtable[words[i]] = 0
                flat_uniwordtable[words[i]] += counts[i]
                uni_count += counts[i]
        return flat_uniwordtable, uni_count

    def init_flat_bigramtable(self, raw_bigramtable):
        flat_bigramtable = {}
        bi_count = 0
        # 遍历raw_bigramtable，统计频率并且转换为扁平化的表
        while raw_bigramtable:
            _, data = raw_bigramtable.popitem() 
            words = data.get("words", [])
            counts = data.get("counts", [])
            for i in range(len(words)):
                parts = words[i].strip().split()
                group = (parts[0], parts[1])
                if group not in flat_bigramtable:
                    flat_bigramtable[group] = 0
                flat_bigramtable[group] += counts[i]
                bi_count += counts[i]
        return flat_bigramtable, bi_count



    # 字典，拼音对应字符列表
    def load_pinyin(self, word2pinyin_file):
        res = {}
        num_of_type = 0
        with open(word2pinyin_file, 'r', encoding='utf-8') as file:
            for line in file:
                num_of_type += 1
                word_and_pinyin = line.strip().split()
                word, pinyin = word_and_pinyin[0], word_and_pinyin[1]
                if pinyin not in res:
                    res[pinyin] = []
                    res[pinyin].append(word)
                else:
                    res[pinyin].append(word)
        return res, num_of_type
    def load_json(self, file):
        with open(file, 'r', encoding='utf-8') as file:
            return json.load(file)
        
    
    def get_uni_count(self):
        return self.uni_count
    
    def get_bi_count(self):
        return self.bi_count
        
class ProbCalculator:
    def __init__(self, corpus, lam=0.98):
        self.corpus = corpus
        # 插值系数，默认为0.98
        self.lam = lam
        # 缓存字典，用于缓存概率
        self.unigram = {}

    def get_unigram(self, word):
        # 如果缓存中存在，则直接返回
        if word in self.unigram:
            return self.unigram[word]
        # 如果没命中，进行计算
        uni_count = self.corpus.get_uni_count()
        count = self.corpus.flat_uniwordtable.get(word, 0)
        # 这里进行了平滑处理，如果字没有出现过的话频数即为0
        # 所以零分子加上1，分母加上总频数，防止出0
        self.unigram[word] = -math.log((count + 1e-7) / (uni_count + 1e-7 * self.corpus.num_of_types))
        return self.unigram[word]
    def get_bigram(self, word1, word2):
        # 内存不够，不做缓存了

        # 1. 计算极大似然估计 P_ML(w2 | w1)
        # count(w1, w2) / count(w1)
        c12 = self.corpus.flat_bigramtable.get((word1, word2), 0)
        c1 = self.corpus.flat_uniwordtable.get(word1, 0)
        p_ml = c12 / c1 if c1 > 0 else 0
        
        # 2. 获取 P(w2) 的概率 
        p_w2 = (self.corpus.flat_uniwordtable.get(word2, 0) + 1e-7) / \
            (self.corpus.get_uni_count() + 1e-7 * self.corpus.num_of_types)
        
        if c1 == 0:
            prob = p_w2
        else:
            # 线性插值
            prob = self.lam * p_ml + (1 - self.lam) * p_w2
        
        return -math.log(prob)
        

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
    first_candidates = corpus.pinyintable.get(pinyinList[0], [])
    first_column = []
    for candidate in first_candidates:
        first_column.append(ViterbiNode(candidate, calculator.get_unigram(candidate), None))
    dp_table.append(first_column)

    # 往后递推
    for i in range(1, len(pinyinList)):
        candidates = corpus.pinyintable.get(pinyinList[i], [])
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

def main():
    # 创建数据类
    corpus = PinyinCorpus('./word2pinyin.txt', './1_word.txt', './2_word.txt')
    calculator = ProbCalculator(corpus, lam=0.9)
    
    # 逐行读取标准输入并进行评测
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
            
        pinyinList = line.split()
        
        # 运行维特比
        result_words = viterbi(pinyinList, corpus, calculator)
        
        # 将列表拼成连续的汉字字符串输出
        print("".join(result_words))

if __name__ == "__main__":
    main()