'''
主函数，这里会导入之前写好的训练模块和解码模块，然后接收拼音输出，输出对应的汉字
支持参数：--model:选择模型 --lam:二元插值参数 --alpha:平滑参数 
'''
from src.train_2g import Corpus2G
from src.model_2g import ParserPinyin2G
from src.train_3g import Corpus3G
from src.model_3g import ParserPinyin3G
import sys
import argparse
import time

# 参数解析类
def parse_args():
    parser = argparse.ArgumentParser(description="拼音输入法程序")

    # 添加模型参数
    # 默认为二元模型
    parser.add_argument(
        "--model",
        type=str,
        default="2g",
        choices=["2g", "3g"],
        help="选择模型类型，默认为二元"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="加法平滑参数"
    )

    parser.add_argument(
        "--lam",
        type=float,
        default=0.9,
        help="二元模型插值参数"
    )

    parser.add_argument(
        "--lam1", 
        type=float, 
        default=0.05, 
        help="三元模型一元回退权重"
        )
    parser.add_argument(
        "--lam2", 
        type=float, 
        default=0.10, 
        help="三元模型二元回退权重"
        )
    parser.add_argument(
        "--lam3", 
        type=float, 
        default=0.85, 
        help="三元模型三元权重"
        )

    return parser.parse_args()


# 因为model中创建二元模型和三元模型的接口不同，所以使用工厂模式创建解码器parser，统一main中调用接口
def build_parser(corpus, args):
    if args.model == "2g":
        # 创建解码器
        return ParserPinyin2G(corpus=corpus, lam=args.lam, alpha=args.alpha)
    elif args.model == "3g":
        return ParserPinyin3G(
            corpus=corpus,
            alpha=args.alpha,
            lam=args.lam,
            lam1=args.lam1,
            lam2=args.lam2,
            lam3=args.lam3
        )
    else:
        raise ValueError("不支持的模型类型")

# 构建二元或三元模型的语料类
def build_corpus(model, corpus_paths, pinyin_path, valid_char_table_path):
    if model == "2g":
        return Corpus2G(
            corpus_path=corpus_paths,
            pinyin_path=pinyin_path,
            valid_char_table_path=valid_char_table_path
        )
    elif model == "3g":
        return Corpus3G(
            corpus_path=corpus_paths,
            pinyin_path=pinyin_path,
            valid_char_table_path=valid_char_table_path
        )
    else:
        raise ValueError("不支持的模型类型")
        


def main():
    # 解析参数
    args = parse_args()
    # 创建语料类
    corpus = build_corpus(
                        model=args.model,
                        corpus_paths=["./corpus/sina_news_gbk"], 
                        pinyin_path="./data/拼音汉字表.txt", 
                        valid_char_table_path="./data/一二级汉字表.txt"
                        )
    
    
    # 创建解码器
    parser = build_parser(
        corpus=corpus,
        args=args
    )
    

    # 读取标准输入
    for pinyin_line in sys.stdin:
        pinyin_line = pinyin_line.strip()

        if not pinyin_line:
            print("")
            continue
        print(parser.parser_pinyin_string(pinyin_line))



if __name__ == "__main__":
    main()