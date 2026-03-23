# 基于字的二元模型的拼音输入法

## 1. 项目简介

本项目实现了一个基于字的二元语言模型的拼音到汉字转换程序。  
程序默认使用字级二元模型，支持命令行输入输出重定向运行，输入输出格式与 OJ 保持一致。

主程序入口为 `main.py`。在默认情况下，可以直接通过以下命令重新生成二元模型在测试集上的输出结果：

```bash
python main.py < data/input.txt > data/output.txt
```

## 2.运行环境

- 操作系统：WSL2
- Python 版本：Python 3
- 第三方依赖：无额外安装包，可直接使用 Python 标准库运行

## 3.项目结构

```
项目根目录/
├─ corpus/                    # 语料库
├─ data/
│  ├─ input.txt              # 测试输入
│  ├─ answer.txt             # 标准答案
│  ├─ output.txt             # 程序输出
│  ├─ 拼音汉字表.txt
│  └─ 一二级汉字表.txt
├─ src/
│  ├─ __init__.py
│  ├─ train_2g.py               # 二元模型训练与频数统计
│  ├─ model_2g.py               # 二元模型概率计算与 Viterbi 解码
│  ├─ model_3g.py               # 三元模型概率计算与 Viterbi 解码 
|  ├─ train_3g.py               # 三元模型训练与频数统计  
│  └─ eval.py                # 本地评测脚本
├─ main.py                   # 主程序入口
├─ requirements.txt
└─ README.md
```

## 4.各文件说明

1. main.py
    主程序入口。负责：
  - 创建语料对象
  - 创建解码器
  - 从标准输入逐行读取拼音
  - 将预测结果输出到标准输出
2. src/train.py
    训练模块。负责：
  - 读取拼音汉字表
  - 读取一二级汉字表
  - 遍历训练语料
  - 统计单字频数和二元频数(如果是3g版本还会统计三元频数)


3. src/model.py
    模型模块。负责：
  - 计算一元概率和二元条件概率（三元概率）
  - 使用计算转移代价
  - 使用 Viterbi 算法寻找最优汉字路径

4. src/eval.py
   评测模块。负责：
  - 读取 data/output.txt
  - 读取 data/answer.txt
  - 计算字准确率和句准确率


### 5.输入输出说明

输入格式

标准输入中每行是一条拼音序列，拼音之间用空格分隔，例如：
```
ren gong zhi neng
bei jing da xue
```

输出格式

标准输出中每行是对应的汉字序列，例如：
```
人工智能
北京大学
```
输入输出格式与 OJ 要求保持一致。

本实验中，拼音表、一二级汉字表以及训练语料为 GBK 编码，因此程序在读取这些文件时显式使用 encoding='gbk'。
程序输出结果文件 data/output.txt 按 UTF-8 编码生成

## 5. 运行方式

本项目的主程序入口为 `main.py`。  
在项目根目录下运行时，程序会默认执行基于字的二元模型，读取标准输入中的拼音序列，并将转换结果输出到标准输出。

### 5.1 默认运行命令

在项目根目录下执行：

```bash
python main.py < data/input.txt > data/output.txt
```
在不添加任何命令行参数时，程序默认执行的是基于字的二元模型

### 5.2 可选参数说明

程序支持以下命令行参数：

```bash
python main.py [-h] [--model {2g,3g}] [--alpha ALPHA] [--lam LAM] [--lam1 LAM1] [--lam2 LAM2] [--lam3 LAM3]
```

参数说明：

- --model {2g,3g}：选择模型类型

  2g：二元模型

  3g：三元模型

  默认通常为二元模型

- --alpha ALPHA：平滑参数

  二元模型和三元模型都可用

- --lam LAM：二元模型中二元概率的插值权重

- --lam1 LAM1：三元模型中一元概率权重

- --lam2 LAM2：三元模型中二元概率权重

- --lam3 LAM3：三元模型中三元概率权重



### 示例

#### 运行二元模型

```
python main.py --model 2g --alpha 0.05 --lam 0.997 < data/input.txt > data/output.txt
```

#### 运行三元模型

```
python main.py --model 3g --alpha 0.05 --lam1 0.05 --lam2 0.20 --lam3 0.75 < data/input.txt > data/output.txt
```

## 6.评测脚本

### 6.1 基础评测

运行基础评测脚本：

```
python src/eval.py
```

基础评测输出：

- 字准确率
- 句准确率

6.2 补充评测

```
python src/extra_eval.py
```

补充评测指标包括：

- CER（字符错误率）
- AED（平均句编辑距离）
- WES（错误句平均错误字数）
- LMR（句长匹配率）
