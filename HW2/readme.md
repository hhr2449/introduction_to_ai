# HW2 文本分类项目说明

本项目实现了一个基于 PyTorch 的二分类文本分类系统，支持 `MLP`、`4 层 MLP`、`TextCNN` 和 `BiLSTM` 四种模型，并支持通过命令行覆盖训练参数、保存实验配置与结果、批量运行多组实验。

## 1. 项目结构

```text
HW2/
├── config/
│   ├── base_config.py          # 公共配置
│   ├── mlp_config.py           # MLP 默认配置
│   ├── cnn_config.py           # TextCNN 默认配置
│   └── rnn_config.py           # RNN/LSTM 默认配置
├── models/
│   ├── mlp.py                  # 普通 MLP
│   ├── mlp_4layers.py          # 4 层 MLP: [512, 256, 128, 64]
│   ├── cnn.py                  # TextCNN
│   └── rnn.py                  # BiLSTM / LSTM
├── train/
│   └── train.py                # 训练入口
├── scripts/
│   ├── run_baselines.sh        # 基线实验
│   ├── run_mlp_experiments.sh  # MLP 参数实验
│   ├── run_textcnn_experiments.sh
│   ├── run_rnn_experiments.sh
│   ├── run_init_experiments.sh # 初始化方法实验
│   └── run_exrta.sh            # 额外实验
├── dataset.py                  # 数据读取、词表构建、DataLoader
├── utils.py                    # 随机种子等工具函数
├── evaluate.py                 # 评估相关文件
├── predict.py                  # 预测相关文件
├── checkpoints/                # 保存最佳模型
├── results/                    # 保存实验配置、训练日志、测试结果
├── run1.sh
├── run2.sh
└── run_all_experiments.sh
```

## 2. 环境依赖

建议使用 Python 3.9 及以上版本，主要依赖包括：

- `torch`
- `numpy`
- `scikit-learn`
- `gensim`
- `tqdm`

如果本地还没有安装，可以手动安装：

```bash
pip install torch numpy scikit-learn gensim tqdm
```

## 3. 数据说明

默认数据路径定义在 [config/base_config.py](/home/hhr/2_spring/IAI/HW2/config/base_config.py:6)：

- `Dataset/train.txt`
- `Dataset/validation.txt`
- `Dataset/test.txt`
- `Dataset/wiki_word2vec_50.bin`

其中：

- 训练、验证、测试文件按行存储样本
- 每行格式为：`label token1 token2 token3 ...`
- 词向量默认使用 50 维预训练词向量

## 4. 支持的模型

当前训练入口 [train/train.py](/home/hhr/2_spring/IAI/HW2/train/train.py:217) 支持以下模型：

- `mlp`：普通 MLP
- `mlp_4layers`：4 层 MLP，隐藏层为 `[512, 256, 128, 64]`
- `textcnn`：TextCNN
- `rnn_lstm`：LSTM / BiLSTM

## 5. 运行方式

### 5.1 单次训练

在 `HW2` 目录下运行：

```bash
python -m train.train --model mlp --exp_name mlp_base
```

例如：

```bash
python -m train.train --model mlp --exp_name mlp_base
python -m train.train --model mlp_4layers --exp_name mlp4_base
python -m train.train --model textcnn --exp_name textcnn_base
python -m train.train --model rnn_lstm --exp_name rnn_base
```

### 5.2 常用命令行参数

所有模型通用参数：

- `--model`
- `--exp_name`
- `--batch_size`
- `--learning_rate`
- `--dropout`
- `--weight_decay`
- `--max_len`
- `--epochs`
- `--patience`
- `--use_early_stopping`
- `--init_method`

TextCNN 专属参数：

- `--num_filters`
- `--filter_sizes`

BiLSTM 专属参数：

- `--hidden_size`
- `--num_layers`
- `--bidirectional`

说明：

- 如果命令行传入参数，则优先使用命令行参数
- 如果命令行未传入，则使用各自 `config/*.py` 中的默认值

### 5.3 参数示例

#### MLP

```bash
python -m train.train \
  --model mlp \
  --exp_name mlp_len100 \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --max_len 100 \
  --epochs 15 \
  --patience 3 \
  --use_early_stopping true \
  --init_method default
```

#### 4 层 MLP

```bash
python -m train.train \
  --model mlp_4layers \
  --exp_name mlp4_xavier \
  --init_method xavier
```

#### TextCNN

```bash
python -m train.train \
  --model textcnn \
  --exp_name textcnn_base \
  --batch_size 64 \
  --learning_rate 1e-3 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --max_len 60 \
  --epochs 15 \
  --patience 3 \
  --use_early_stopping true \
  --num_filters 100 \
  --filter_sizes 3 4 5 \
  --init_method default
```

#### BiLSTM

```bash
python -m train.train \
  --model rnn_lstm \
  --exp_name rnn_base \
  --batch_size 32 \
  --learning_rate 5e-4 \
  --dropout 0.5 \
  --weight_decay 1e-4 \
  --max_len 60 \
  --epochs 15 \
  --patience 3 \
  --use_early_stopping true \
  --hidden_size 128 \
  --num_layers 1 \
  --bidirectional true \
  --init_method default
```

## 6. 初始化方法

`--init_method` 当前支持：

- `default`
- `xavier`
- `kaiming`
- `orthogonal`
- `normal`

说明：

- `default` 表示使用 PyTorch 默认初始化
- 其他初始化方式会作用在 `Linear`、`Conv` 和 `LSTM/RNN/GRU` 等模块上

## 7. 实验结果保存

每次训练会自动创建实验目录：

```text
results/<model>/<exp_name>/
```

其中保存：

- `config.json`：本次实验的关键参数
- `train_log.csv`：每个 epoch 的训练/验证指标
- `test_metrics.json`：最终测试集结果

最佳模型会保存到：

```text
checkpoints/<model>/<exp_name>/best_model.pt
```

### 7.1 `train_log.csv` 字段

- `epoch`
- `train_loss`
- `train_acc`
- `valid_loss`
- `valid_acc`
- `valid_precision`
- `valid_recall`
- `valid_f1`

### 7.2 `test_metrics.json` 字段

- `best_valid_f1`
- `test_loss`
- `test_acc`
- `test_precision`
- `test_recall`
- `test_f1`

## 8. 批量运行脚本

项目中已经提供了一些批量实验脚本：

- [scripts/run_baselines.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_baselines.sh:1)
  用于运行基础对比实验
- [scripts/run_mlp_experiments.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_mlp_experiments.sh:1)
  用于运行 MLP 的 batch size、learning rate、dropout、max length 实验
- [scripts/run_textcnn_experiments.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_textcnn_experiments.sh:1)
  用于运行 TextCNN 的学习率、卷积核大小、卷积核数量等实验
- [scripts/run_rnn_experiments.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_rnn_experiments.sh:1)
  用于运行 RNN/LSTM 的 hidden size、层数、单双向等实验
- [scripts/run_init_experiments.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_init_experiments.sh:1)
  用于比较不同初始化方法
- [scripts/run_exrta.sh](/home/hhr/2_spring/IAI/HW2/scripts/run_exrta.sh:1)
  用于运行附加实验

运行示例：

```bash
bash scripts/run_baselines.sh
bash scripts/run_mlp_experiments.sh
bash scripts/run_textcnn_experiments.sh
bash scripts/run_rnn_experiments.sh
bash scripts/run_init_experiments.sh
bash scripts/run_exrta.sh
```

也可以通过总脚本运行：

```bash
bash run2.sh
```

## 9. 默认配置

公共默认参数定义在 [config/base_config.py](/home/hhr/2_spring/IAI/HW2/config/base_config.py:1) 中，主要包括：

- `BATCH_SIZE = 64`
- `LEARNING_RATE = 1e-3`
- `EPOCHS = 20`
- `DROPOUT = 0.5`
- `WEIGHT_DECAY = 1e-4`
- `MAX_SENTENCE_LEN = 120`
- `USE_EARLY_STOPPING = True`
- `PATIENCE = 3`
- `INIT_METHOD = "default"`

不同模型的专属默认参数分别在：

- [config/mlp_config.py](/home/hhr/2_spring/IAI/HW2/config/mlp_config.py:1)
- [config/cnn_config.py](/home/hhr/2_spring/IAI/HW2/config/cnn_config.py:1)
- [config/rnn_config.py](/home/hhr/2_spring/IAI/HW2/config/rnn_config.py:1)

## 10. 说明

- 项目默认使用验证集 `F1` 作为最佳模型保存依据
- 若开启早停，则当验证集 `F1` 在连续若干轮内不提升时提前结束训练
- 训练完成后会自动加载最佳模型并在测试集上评估
