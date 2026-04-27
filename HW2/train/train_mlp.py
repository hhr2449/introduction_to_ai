import os
import torch
import torch.nn as nn

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import get_data_loaders
from models.mlp import MLP
from config.base_config import (
    DEVICE,
    SEED,
    PAD_TOKEN,
)
from config.mlp_config import (
    LEARNING_RATE,
    EPOCHS,
    CHECKPOINT_PATH,
    WEIGHT_DECAY,
    USE_EARLY_STOPPING,
    PATIENCE,
)
from HW2.utils import set_seed


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


def main():
    set_seed(SEED)
    # 创建保存最佳模型路径
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    # 获取DataLoader
    train_loader, valid_loader, test_loader, word2id, _, embedding_matrix = get_data_loaders()
    # 创建模型实例
    model = MLP(
        vocab_size=len(word2id),
        pad_idx=word2id[PAD_TOKEN],
        embedding_matrix=embedding_matrix,
        freeze_embedding=False,
    ).to(DEVICE)
    # 创建损失函数和优化器
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 记录当前最优测试集f1值
    # 考虑每进行一次测试就进行一次验证，如果验证集f1值更高，则保存模型
    best_f1 = 0.0
    # 早停计数器
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = evaluate(model, valid_loader, criterion)
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            f" | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}"
            f" | Valid Precision: {valid_precision:.4f} | Valid Recall: {valid_recall:.4f}"
            f" | Valid F1: {valid_f1:.4f}"
        )
        # 验证集f1值更高，则保存模型，同时更新早停计数器
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            patience_counter = 0
            print(f"Saved model with valid f1: {valid_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Early stop counter: {patience_counter}/{PATIENCE}")
            if USE_EARLY_STOPPING and patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break
    
    # 训练完成
    print("Finished training.")
    print(f"Best Valid f1: {best_f1:.4f}")
    # 在测试集上测试
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
          f" | Test Precision: {test_precision:.4f} | Test Recall: {test_recall:.4f}"
          f" | Test F1: {test_f1:.4f}")

    


if __name__ == "__main__":
    main()
