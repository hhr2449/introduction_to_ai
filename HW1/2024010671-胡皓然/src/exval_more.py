import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # 删除
                dp[j - 1] + 1,  # 插入
                prev + cost     # 替换/匹配
            )
            prev = tmp
    return dp[m]

def extra_evaluate(output_path=None, answer_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if output_path is None:
        output_path = os.path.join(base_dir,'data', 'output.txt')
    if answer_path is None:
        answer_path = os.path.join(base_dir,'data', 'answer.txt')

    outputs = read_file(output_path)
    answers = read_file(answer_path)

    total_chars = 0
    total_edit = 0
    total_sent = 0
    wrong_sent = 0
    wrong_sent_edit_sum = 0
    length_match = 0

    for pred, gold in zip(outputs, answers):
        total_sent += 1
        total_chars += len(gold)

        ed = levenshtein(pred, gold)
        total_edit += ed

        if len(pred) == len(gold):
            length_match += 1

        if pred != gold:
            wrong_sent += 1
            wrong_sent_edit_sum += ed

    cer = total_edit / total_chars if total_chars else 0.0
    aed = total_edit / total_sent if total_sent else 0.0
    wes = wrong_sent_edit_sum / wrong_sent if wrong_sent else 0.0
    lmr = length_match / total_sent if total_sent else 0.0

    return {
        "CER": cer,
        "AED": aed,
        "WES": wes,
        "LMR": lmr,
        "total_edit": total_edit,
        "total_sent": total_sent,
        "wrong_sent": wrong_sent,
    }

def main():
    metrics = extra_evaluate()
    print(f"CER(字符错误率): {metrics['CER'] * 100:.4f}%")
    print(f"AED(平均句编辑距离): {metrics['AED']:.4f}")
    print(f"WES(错误句平均错误字数): {metrics['WES']:.4f}")
    print(f"LMR(句长匹配率): {metrics['LMR'] * 100:.4f}%")
    print(f"总编辑距离: {metrics['total_edit']}")
    print(f"总句数: {metrics['total_sent']}")
    print(f"错误句数: {metrics['wrong_sent']}")

if __name__ == "__main__":
    main()