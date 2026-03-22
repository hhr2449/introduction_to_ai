'''
用于二元模型
1.读取汉字表，生成拼音和汉字的对应字典
2.读取训练语料
3.对训练语料进行清洗，去除多余的符号
4.统计词频，使用字典进行保存
5.将生成的这些数据返回
'''

import os


# 使用类进行封装，将语料信息封装在类中，训练完后直接返回对象以供使用
class Corpus2G:
    # 输入语料库路径，拼音表路径
    def __init__(self, corpus_path, pinyin_path, valid_char_table_path):
        # 关键的数据表：拼音汉字对照表，合法汉字表，单字频数表，双字频数表
        self.pinyin_table, self.valid_char_table = self.load_pinyin_table(pinyin_path, valid_char_table_path)
        self.uniword_table, self.bigram_table = self.calculate_frequency_corpus(corpus_path)
        # 统计一元字符总频率，二元字符总频率，出现过的字符总类数
        self.uni_count = sum(self.uniword_table.values())
        self.bi_count = sum(self.bigram_table.values())
        self.num_of_types = len(self.uniword_table)

    # 兼容utf-8和gbk，会先尝试，若失败则切换
    # 读取为大字符串
    def read_text_auto(self, file_path):
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    # 按行读取成列表
    def read_lines_auto(self, file_path):
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()


    # 读取拼音汉字表并且存储在表中
    # 参数：拼音表路径，有效的字符表路径,在读取拼音表的时候需要注意只读入有效字符
    def load_pinyin_table(self, pinyin_path, valid_char_table_path):
        # 先获取有效字符表
        # 汉字表中为连续的汉字，只需要全部读入然后使用set()方法就可以获取集合
        # 直接读成一大串字符串
        all_text = self.read_text_auto(valid_char_table_path)
        valid_char_table = set(all_text)
        # 解析拼音表
        # 解析成{'拼音':['字1','字2',...],'拼音':['字1','字2',...],...}
        pinyin_table = {}
        # 解析成行的列表
        all_lines_list = self.read_lines_auto(pinyin_path)
        # 一行行的读入
        for line in all_lines_list:
            # 直接分割成一个列表
            one_line_list = line.strip().split()
            # 第一个元素是拼音
            pinyin = one_line_list[0]
            char_list = []
            # 检查是否是合法的
            for char in one_line_list[1:]:
                if char in valid_char_table:
                    char_list.append(char)
            # 将拼音和汉字列表加入到字典中
            pinyin_table[pinyin] = char_list
        return pinyin_table, valid_char_table

    # 接下来进行词频的统计

    # 直接在一个函数中统计完一元和二元的频数
    # 输入的是存放语料的文件夹路径，函数中会自动遍历文件
    def calculate_frequency_corpus(self, file_path_list):
        uniword_table = {}
        bigram_table = {}
        for file_path in file_path_list:
            # 遍历语料文件夹中的所有文件
            for root, _, files in os.walk(file_path):
                for file in files:
                    # 获取文件路径
                    full_file_path = os.path.join(root, file)
                    # 打开文件
                    all_lines_list = self.read_lines_auto(full_file_path)
                    # 前一个字符
                    prev = None
                    
                    # 逐行读取
                    for line in all_lines_list:
                        for char in line:
                            if char in self.valid_char_table:
                                # 一元
                                if char not in uniword_table:
                                    uniword_table[char] = 0
                                uniword_table[char] += 1
                                # 二元
                                if prev != None:
                                    char_pair = (prev, char)
                                    if char_pair not in bigram_table:
                                        bigram_table[char_pair] = 0
                                    bigram_table[char_pair] += 1
                                # 窗口移动
                                prev = char
                            else:
                                # 防止跨非法字符连接
                                prev = None
        return uniword_table, bigram_table
