import csv
import json
import os
import torch
import torch.nn as nn
import argparse
import torch.nn.init as init

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import get_data_loaders
from models.mlp import MLP
from models.mlp_4layers import MLP as MLP4Layers
from models.cnn import TextCNN
from models.rnn import RNN_LSTM
from config.base_config import (
    CHECKPOINT_DIR,
    DEVICE,
    RESULT_DIR,
    SEED,
    PAD_TOKEN,
)

from utils import set_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def override_config_from_args(args, config):
    common_overrides = {
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.learning_rate,
        "DROPOUT": args.dropout,
        "WEIGHT_DECAY": args.weight_decay,
        "MAX_SENTENCE_LEN": args.max_len,
        "EPOCHS": args.epochs,
        "PATIENCE": args.patience,
        "USE_EARLY_STOPPING": args.use_early_stopping,
    }
    for key, value in common_overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.init_method is not None:
        setattr(config, "INIT_METHOD", args.init_method)

    if args.model == "textcnn":
        if args.num_filters is not None:
            config.NUM_FILTERS = args.num_filters
        if args.filter_sizes is not None:
            config.FILTER_SIZES = args.filter_sizes

    if args.model == "rnn_lstm":
        if args.hidden_size is not None:
            config.HIDDEN_SIZE = args.hidden_size
        if args.num_layers is not None:
            config.NUM_LAYERS = args.num_layers
        if args.bidirectional is not None:
            config.BIDIRECTIONAL = args.bidirectional

    return config


def init_module_weights(module, init_method):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if init_method == "xavier":
            init.xavier_uniform_(module.weight)
        elif init_method == "kaiming":
            init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif init_method == "orthogonal":
            init.orthogonal_(module.weight)
        elif init_method == "normal":
            init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        for name, param in module.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                if init_method == "xavier":
                    init.xavier_uniform_(param)
                elif init_method == "kaiming":
                    init.kaiming_uniform_(param, nonlinearity="relu")
                elif init_method == "orthogonal":
                    init.orthogonal_(param)
                elif init_method == "normal":
                    init.normal_(param, mean=0.0, std=0.02)
            elif "bias" in name:
                init.zeros_(param)


def apply_initialization(model, init_method):
    if init_method == "default":
        return
    model.apply(lambda module: init_module_weights(module, init_method))

        
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    # 切换训练模式
    model.train()

    # 保存总的损失和正确率
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 开始迭代data_loader
    # tqdm用于显示进度
    for input_ids, labels in tqdm(data_loader, desc="Training", leave=False):
        # 将tensor移动到设备
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 清空梯度
        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        logits = model(input_ids)

        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size
    
    # 完成一轮训练，计算平均损失和正确率，返回
    return total_loss / total_samples, total_correct / total_samples

# 验证
# 不进行梯度更新
def evaluate(model, data_loader, criterion):
    # 开启验证模式
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    # 搜集所有的预测结果和真实标签
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for input_ids, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            # 前向传播，获取计算结果
            logits = model(input_ids)
            # 计算损失
            loss = criterion(logits, labels)
            # labels的0维大小，即为batch_size
            batch_size = labels.size(0)
            # loss是损失的平均值，需要乘以batch_size
            total_loss += loss.item() * batch_size
            # 第1维即为两个分类的预测值，取其中较大的作为预测结果
            preds = logits.argmax(dim=1)
            # 计算预测结果和真实标签相同的样本数
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, avg_acc, precision, recall, f1


# 根据参数构建实验配置
def build_experiment_config(args, config):
    experiment_config = {
        "model": args.model,
        "exp_name": args.exp_name,
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "batch_size": config.BATCH_SIZE,
        "max_len": config.MAX_SENTENCE_LEN,
        "patience": config.PATIENCE,
        "use_early_stopping": config.USE_EARLY_STOPPING,
        "init_method": getattr(config, "INIT_METHOD", "default"),
    }
    if args.model == "textcnn":
        experiment_config["num_filters"] = config.NUM_FILTERS
        experiment_config["filter_sizes"] = config.FILTER_SIZES
    if args.model == "rnn_lstm":
        experiment_config["hidden_size"] = config.HIDDEN_SIZE
        experiment_config["num_layers"] = config.NUM_LAYERS
        experiment_config["bidirectional"] = config.BIDIRECTIONAL
    return experiment_config


def save_json(file_path, payload):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    # 选择训练参数
    parser = argparse.ArgumentParser()
    # 选择模型
    parser.add_argument("--model", type=str, required=True, choices=["mlp", "mlp_4layers", "textcnn", "rnn_lstm"])
    # 实验名称
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--use_early_stopping", type=str2bool, default=None)
    parser.add_argument("--num_filters", type=int, default=None)
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--bidirectional", type=str2bool, default=None)
    parser.add_argument(
        "--init_method",
        type=str,
        choices=["default", "xavier", "kaiming", "orthogonal", "normal"],
        default=None,
    )
    args = parser.parse_args()
    print(f"Training model: {args.model}")
    print(f"Experiment name: {args.exp_name}")

    set_seed(SEED)

    # 创建模型实例
    # 根据参数来进行模型创建
    if args.model == "mlp":
        import config.mlp_config as config
    elif args.model == "mlp_4layers":
        import config.mlp_4layers_config as config
    elif args.model == "textcnn":
        import config.cnn_config as config
    elif args.model == "rnn_lstm":
        import config.rnn_config as config
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    config = override_config_from_args(args, config)

    # 实验结果保存路径
    experiment_dir = os.path.join(RESULT_DIR, args.model, args.exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    # 保存训练日志、配置文件、测试结果
    train_log_path = os.path.join(experiment_dir, "train_log.csv")
    config_path = os.path.join(experiment_dir, "config.json")
    test_metrics_path = os.path.join(experiment_dir, "test_metrics.json")
    # 最佳模型保存路径
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, args.model, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    experiment_config = build_experiment_config(args, config)
    save_json(config_path, experiment_config)

    # 获取DataLoader
    train_loader, valid_loader, test_loader, word2id, _, embedding_matrix = get_data_loaders(
        batch_size=config.BATCH_SIZE,
        max_len=config.MAX_SENTENCE_LEN,
    )

    if args.model == "mlp":
        model = MLP(
            vocab_size=len(word2id),
            pad_idx=word2id[PAD_TOKEN],
            embedding_matrix=embedding_matrix,
            freeze_embedding=False,
            dropout=config.DROPOUT,
        ).to(DEVICE)
    elif args.model == "mlp_4layers":
        model = MLP4Layers(
            vocab_size=len(word2id),
            pad_idx=word2id[PAD_TOKEN],
            embedding_matrix=embedding_matrix,
            freeze_embedding=False,
            dropout=config.DROPOUT,
        ).to(DEVICE)
    elif args.model == "textcnn":
        model = TextCNN(
            vocab_size=len(word2id),
            pad_idx=word2id[PAD_TOKEN],
            embedding_matrix=embedding_matrix,
            freeze_embedding=False,
            num_filters=config.NUM_FILTERS,
            filter_sizes=config.FILTER_SIZES,
            dropout=config.DROPOUT,
        ).to(DEVICE)
    elif args.model == "rnn_lstm":
        model = RNN_LSTM(
            vocab_size=len(word2id),
            pad_idx=word2id[PAD_TOKEN],
            embedding_matrix=embedding_matrix,
            freeze_embedding=False,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
        ).to(DEVICE)
    apply_initialization(model, getattr(config, "INIT_METHOD", "default"))

    # 创建损失函数和优化器
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # 记录当前最优测试集f1值
    # 考虑每进行一次测试就进行一次验证，如果验证集f1值更高，则保存模型
    best_f1 = float("-inf")
    # 早停计数器
    patience_counter = 0
    train_log_rows = []
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion)
        train_log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
                "valid_precision": valid_precision,
                "valid_recall": valid_recall,
                "valid_f1": valid_f1,
            }
        )
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            f" | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}"
            f" | Valid Precision: {valid_precision:.4f} | Valid Recall: {valid_recall:.4f}"
            f" | Valid F1: {valid_f1:.4f}"
        )
        # 验证集f1值更高，则保存模型，同时更新早停计数器
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
            print(f"Saved model with valid f1: {valid_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop counter: {patience_counter}/{config.PATIENCE}")
            if config.USE_EARLY_STOPPING and patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break
    # 保存训练日志
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "valid_loss",
                "valid_acc",
                "valid_precision",
                "valid_recall",
                "valid_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(train_log_rows)
    
    # 训练完成
    print("Finished training.")
    print(f"Best Valid f1: {best_f1:.4f}")
    # 在测试集上测试
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
          f" | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}"
          f" | Test F1: {test_f1:.4f}")
    # 保存测试结果
    save_json(
        test_metrics_path,
        {
            "best_valid_f1": best_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        },
    )

    


if __name__ == "__main__":
    main()
