'''
主函数，这里会导入之前写好的训练模块和解码模块，然后接收拼音输出，输出对应的汉字
'''
from src.train import Corpus
from src.model import ParserPinyin
import sys

def main():
    # 创建语料类
    corpus = Corpus(corpus_path="./corpus", 
                    pinyin_path="./data/拼音汉字表.txt", 
                    valid_char_table_path="./data/一二级汉字表.txt")
    

    # 创建解码器
    parser = ParserPinyin(corpus=corpus, lam=0.9, alpha=0.1)

    # 读取标准输入
    for pinyin_line in sys.stdin:
        pinyin_line = pinyin_line.strip()

        if not pinyin_line:
            print("")
            continue

        print(parser.parser_pinyin_string(pinyin_line))


if __name__ == "__main__":
    main()