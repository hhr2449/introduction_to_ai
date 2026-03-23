import argparse
import csv
import hashlib
import itertools
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable, List

from train_2g import Corpus2G
from train_3g import Corpus3G
from model_2g import ParserPinyin2G
from model_3g import ParserPinyin3G


def parse_float_list(text: str) -> List[float]:
    if not text:
        return []
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def parse_path_list(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(',') if x.strip()]


def read_lines(path: Path, encoding: str = 'utf-8') -> List[str]:
    with path.open('r', encoding=encoding) as f:
        return [line.rstrip('\n') for line in f]


def evaluate(output_list: List[str], answer_list: List[str]):
    correct_chars = 0
    total_chars = 0
    correct_sens = 0
    total_sens = 0

    for output_sen, answer_sen in zip(output_list, answer_list):
        compare_len = min(len(output_sen), len(answer_sen))
        total_chars += len(answer_sen)
        for i in range(compare_len):
            if output_sen[i] == answer_sen[i]:
                correct_chars += 1

        total_sens += 1
        if len(output_sen) == len(answer_sen) and output_sen == answer_sen:
            correct_sens += 1

    char_acc = correct_chars / total_chars if total_chars else 0.0
    sen_acc = correct_sens / total_sens if total_sens else 0.0
    return char_acc, sen_acc


def build_grid(args):
    rows = []
    if args.model == '2g':
        for alpha, lam in itertools.product(args.alphas, args.lams):
            rows.append({
                'model': '2g',
                'alpha': alpha,
                'lam': lam,
                'lam1': '',
                'lam2': '',
                'lam3': '',
            })
    else:
        for alpha, lam, lam1, lam2, lam3 in itertools.product(
            args.alphas, args.lams, args.lam1s, args.lam2s, args.lam3s
        ):
            lam_sum = lam1 + lam2 + lam3
            if args.require_sum1 and abs(lam_sum - 1.0) > args.sum_tol:
                continue
            rows.append({
                'model': '3g',
                'alpha': alpha,
                'lam': lam,
                'lam1': lam1,
                'lam2': lam2,
                'lam3': lam3,
            })
    return rows


def make_cache_key(model: str, corpus_dirs: Iterable[str], pinyin_path: str, valid_char_path: str) -> str:
    raw = '|'.join([model, *sorted(corpus_dirs), pinyin_path, valid_char_path])
    return hashlib.md5(raw.encode('utf-8')).hexdigest()


def train_or_load_corpus(args, project_root: Path):
    corpus_dirs = [str((project_root / p).resolve()) if not Path(p).is_absolute() else p for p in args.corpus_dirs]
    pinyin_path = str((project_root / args.pinyin_path).resolve()) if not Path(args.pinyin_path).is_absolute() else args.pinyin_path
    valid_char_path = str((project_root / args.valid_char_path).resolve()) if not Path(args.valid_char_path).is_absolute() else args.valid_char_path

    cache_info = {'cache_hit': False, 'cache_path': None}
    corpus = None
    train_time_s = None

    if args.use_pickle_cache:
        cache_dir = (project_root / args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = make_cache_key(args.model, corpus_dirs, pinyin_path, valid_char_path)
        cache_path = cache_dir / f'corpus_{cache_key}.pkl'
        cache_info['cache_path'] = str(cache_path)
        if cache_path.exists():
            with cache_path.open('rb') as f:
                payload = pickle.load(f)
            corpus = payload['corpus']
            train_time_s = payload.get('train_time_s', 0.0)
            cache_info['cache_hit'] = True

    if corpus is None:
        start = time.perf_counter()
        if args.model == '2g':
            corpus = Corpus2G(corpus_dirs, pinyin_path, valid_char_path)
        else:
            corpus = Corpus3G(corpus_dirs, pinyin_path, valid_char_path)
        train_time_s = time.perf_counter() - start

        if args.use_pickle_cache:
            with Path(cache_info['cache_path']).open('wb') as f:
                pickle.dump({'corpus': corpus, 'train_time_s': train_time_s}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return corpus, train_time_s, cache_info


def predict_lines(parser, input_lines: List[str]) -> List[str]:
    outputs = []
    for line in input_lines:
        stripped = line.strip()
        if not stripped:
            outputs.append('')
            continue
        outputs.append(parser.parser_pinyin_string(stripped))
    return outputs


def run_one(corpus, train_info: dict, row: dict, input_lines: List[str], answer_lines: List[str], save_pred_path: Path | None):
    if row['model'] == '2g':
        parser = ParserPinyin2G(corpus, lam=row['lam'], alpha=row['alpha'])
    else:
        parser = ParserPinyin3G(
            corpus,
            alpha=row['alpha'],
            lam=row['lam'],
            lam1=row['lam1'],
            lam2=row['lam2'],
            lam3=row['lam3'],
        )

    start = time.perf_counter()
    outputs = predict_lines(parser, input_lines)
    predict_time_s = time.perf_counter() - start

    if save_pred_path is not None:
        save_pred_path.parent.mkdir(parents=True, exist_ok=True)
        with save_pred_path.open('w', encoding='utf-8') as f:
            for line in outputs:
                f.write(line + '\n')

    char_acc, sen_acc = evaluate(outputs, answer_lines)

    result = dict(row)
    result.update({
        'char_acc': char_acc,
        'sen_acc': sen_acc,
        'char_acc_pct': char_acc * 100,
        'sen_acc_pct': sen_acc * 100,
        'train_time_s': train_info['train_time_s'],
        'predict_time_s': predict_time_s,
        'uni_items': len(corpus.uniword_table),
        'bi_items': len(corpus.bigram_table),
        'tri_items': len(corpus.trigram_table) if hasattr(corpus, 'trigram_table') else '',
        'uni_count': corpus.uni_count,
        'bi_count': corpus.bi_count,
        'tri_count': getattr(corpus, 'tri_count', ''),
        'cache_hit': train_info['cache_hit'],
    })
    return result


def make_pred_filename(row: dict) -> str:
    if row['model'] == '2g':
        return f"2g_a{row['alpha']}_l{row['lam']}.txt"
    return (
        f"3g_a{row['alpha']}_l{row['lam']}_"
        f"l1_{row['lam1']}_l2_{row['lam2']}_l3_{row['lam3']}.txt"
    )


def main():
    parser = argparse.ArgumentParser(description='快速自动调参脚本：词频表只训练一次，后续重复预测')
    parser.add_argument('--project-root', type=str, default='.', help='项目根目录')
    parser.add_argument('--model', type=str, default='3g', choices=['2g', '3g'], help='选择调参模型')

    # 数据路径
    parser.add_argument('--input', type=str, default='data/input.txt', help='测试输入文件')
    parser.add_argument('--answer', type=str, default='data/answer.txt', help='标准答案文件')
    parser.add_argument('--corpus-dirs', type=parse_path_list, required=True, help='训练语料目录，多个目录用逗号分隔')
    parser.add_argument('--pinyin-path', type=str, required=True, help='拼音汉字表路径')
    parser.add_argument('--valid-char-path', type=str, required=True, help='一二级汉字表路径')
    parser.add_argument('--output-csv', type=str, default='tuning_results_fast.csv', help='结果CSV文件')

    # 是否缓存训练好的语料对象
    parser.add_argument('--use-pickle-cache', action='store_true', help='将训练好的语料对象缓存到磁盘，下次可直接加载')
    parser.add_argument('--cache-dir', type=str, default='.tune_cache', help='pickle缓存目录')

    # 是否保存每组参数的输出结果
    parser.add_argument('--save-predictions', action='store_true', help='保存每组参数对应的预测结果文件')
    parser.add_argument('--pred-dir', type=str, default='tune_outputs', help='预测结果保存目录')

    # 参数网格
    parser.add_argument('--alphas', type=parse_float_list, default=parse_float_list('0.1'), help='例如 0.01,0.05,0.1')
    parser.add_argument('--lams', type=parse_float_list, default=parse_float_list('0.9'), help='例如 0.85,0.9,0.95')
    parser.add_argument('--lam1s', type=parse_float_list, default=parse_float_list('0.05'), help='例如 0.02,0.05,0.1')
    parser.add_argument('--lam2s', type=parse_float_list, default=parse_float_list('0.1'), help='例如 0.08,0.1,0.2')
    parser.add_argument('--lam3s', type=parse_float_list, default=parse_float_list('0.85'), help='例如 0.7,0.75,0.85,0.9')
    parser.add_argument('--require-sum1', action='store_true', help='仅保留 lam1+lam2+lam3≈1 的组合')
    parser.add_argument('--sum-tol', type=float, default=1e-9, help='权重和为1时的误差容忍')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_path = (project_root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    answer_path = (project_root / args.answer).resolve() if not Path(args.answer).is_absolute() else Path(args.answer)
    output_csv = (project_root / args.output_csv).resolve() if not Path(args.output_csv).is_absolute() else Path(args.output_csv)
    pred_dir = (project_root / args.pred_dir).resolve() if not Path(args.pred_dir).is_absolute() else Path(args.pred_dir)

    if not input_path.exists():
        raise FileNotFoundError(f'找不到输入文件: {input_path}')
    if not answer_path.exists():
        raise FileNotFoundError(f'找不到答案文件: {answer_path}')

    for p in args.corpus_dirs:
        path = (project_root / p).resolve() if not Path(p).is_absolute() else Path(p)
        if not path.exists():
            raise FileNotFoundError(f'找不到训练语料目录: {path}')

    pinyin_check = (project_root / args.pinyin_path).resolve() if not Path(args.pinyin_path).is_absolute() else Path(args.pinyin_path)
    valid_check = (project_root / args.valid_char_path).resolve() if not Path(args.valid_char_path).is_absolute() else Path(args.valid_char_path)
    if not pinyin_check.exists():
        raise FileNotFoundError(f'找不到拼音表文件: {pinyin_check}')
    if not valid_check.exists():
        raise FileNotFoundError(f'找不到汉字表文件: {valid_check}')

    input_lines = read_lines(input_path, encoding='utf-8')
    answer_lines = read_lines(answer_path, encoding='utf-8')
    if len(input_lines) != len(answer_lines):
        print(f'警告：输入行数({len(input_lines)})与答案行数({len(answer_lines)})不一致，评测将按最短长度对齐。', file=sys.stderr)

    grid = build_grid(args)
    if not grid:
        raise ValueError('参数组合为空，请检查输入范围')

    print(f'共需测试 {len(grid)} 组参数。')
    print('正在训练/加载语料对象...')
    corpus, train_time_s, cache_info = train_or_load_corpus(args, project_root)
    print(
        f"语料已就绪：训练时间={train_time_s:.4f}s | "
        f"单字={len(corpus.uniword_table)} | 二元={len(corpus.bigram_table)}"
        + (f" | 三元={len(corpus.trigram_table)}" if hasattr(corpus, 'trigram_table') else '')
        + (f" | cache_hit={cache_info['cache_hit']}" if args.use_pickle_cache else '')
    )

    train_info = {
        'train_time_s': train_time_s,
        'cache_hit': cache_info['cache_hit'],
    }

    fieldnames = [
        'model', 'alpha', 'lam', 'lam1', 'lam2', 'lam3',
        'char_acc', 'sen_acc', 'char_acc_pct', 'sen_acc_pct',
        'train_time_s', 'predict_time_s',
        'uni_items', 'bi_items', 'tri_items',
        'uni_count', 'bi_count', 'tri_count',
        'cache_hit',
    ]

    results = []
    for i, row in enumerate(grid, start=1):
        print(f"[{i}/{len(grid)}] 测试参数: {row}")
        save_pred_path = pred_dir / make_pred_filename(row) if args.save_predictions else None
        result = run_one(corpus, train_info, row, input_lines, answer_lines, save_pred_path)
        results.append(result)
        print(
            f"  字准确率={result['char_acc_pct']:.4f}% | "
            f"句准确率={result['sen_acc_pct']:.4f}% | "
            f"预测={result['predict_time_s']:.4f}s"
        )

    results.sort(key=lambda x: (x['sen_acc'], x['char_acc']), reverse=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    best = results[0]
    print('\n最优参数（按句准确率优先、字准确率次之排序）:')
    print(best)
    print(f'结果已保存到: {output_csv}')


if __name__ == '__main__':
    main()
