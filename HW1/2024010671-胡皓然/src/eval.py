'''
比较程序输出和标准输出，得出字准确率和句准确率
'''


# read_file():读入一个文件地址，然后按照行进行划分，读成一个字符串列表
# calc_char_accuracy()：读入两个字符串列表，比较字准确率。总字数为正确版本字数
# calc_sentence_accuracy()：读入两个文件列表，比较句准确率

import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        res = []
        for line in f:
            res.append(line.rstrip('\n'))
        return res

def calculate_char_accuracy(output_list, answer_list):
    correct_chars = 0
    total_chars = 0

    # 遍历两个字符串列表
    # 使用zip进行遍历
    for output_sen, answer_sen in zip(output_list, answer_list):
        # 就短
        compare_len = min(len(output_sen), len(answer_sen))

        # 总字符数由正确的句子决定
        total_chars += len(answer_sen)
        for i in range(compare_len):
            if output_sen[i] == answer_sen[i]:
                correct_chars += 1
        
    
    if total_chars == 0:
        return 0.0
    else:
        return correct_chars / total_chars

def calculate_sen_accuracy(output_list, answer_list):
    correct_sens = 0
    total_sens = 0

    for output_sen, answer_sen in zip(output_list, answer_list):
        
        total_sens += 1
        # 长度不同直接否决
        if len(output_sen) != len(answer_sen):
            continue
        
        if output_sen == answer_sen:
            correct_sens += 1
    
    if total_sens == 0:
        return 0.0
    else:
        return correct_sens / total_sens

def evaluate(output_path=None, answer_path=None):
    # 项目根路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 拼接出路径
    if output_path is None:
        output_path = os.path.join(base_dir, 'data', 'output.txt')
    if answer_path is None:
        answer_path = os.path.join(base_dir, 'data', 'answer.txt')
    # 先转换成列表
    output_list = read_file(output_path)
    answer_list = read_file(answer_path)

    # 调用函数计算准确率
    char_acc = calculate_char_accuracy(output_list, answer_list)
    sen_acc = calculate_sen_accuracy(output_list, answer_list)

    return char_acc, sen_acc

def main():
    char_acc, sen_acc = evaluate()
    print(f"字准确率: {char_acc * 100}%")
    print(f"句准确率: {sen_acc * 100}%")

if __name__ == "__main__":
    main()