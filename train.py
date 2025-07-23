import os
import random
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from networks.fdrl import FDRL


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="exp_1", help="实验名称")
    parser.add_argument("--raf_path", type=str, default="datasets/raf-basic/", help="Raf-DB 数据集路径")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam 优化器初始学习率")
    parser.add_argument("--workers", default=4, type=int, help="数据加载线程数")
    parser.add_argument("--epochs", type=int, default=40, help="训练总轮数")
    parser.add_argument("--num_class", type=int, default=7, help="类别数")
    parser.add_argument("--num_branch", type=int, default=9, help="分支网络数")
    parser.add_argument("--feat_dim", type=int, default=128, help="特征维度")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重文件路径")
    return parser.parse_args()


class RafDataSet(data.Dataset):
    """处理 Raf-DB 数据集"""

    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        label_file = os.path.join(raf_path, f"EmoLabel/{phase}_list.txt")
        df = pd.read_csv(label_file, sep=" ", header=None, names=["name", "label"])
        file_names = df["name"].values
        self.label = df["label"].values - 1
        self.file_paths = [os.path.join(self.raf_path, "Image/aligned", f.split(".")[0] + "_aligned.jpg") for f in
                           file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.label[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def plot_metrics(train_losses, train_accs, val_losses, val_accs, exp_name):
    """绘制训练和验证的损失和准确率曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'./metrics_{exp_name}.png')
    plt.close()


def run_training():
    """训练函数"""
    args = parse_args()
    setup_seed(456)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = FDRL(args.num_branch, 512, args.feat_dim, args.num_class)
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = RafDataSet(args.raf_path, phase="train", transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, pin_memory=True)

    val_dataset = RafDataSet(args.raf_path, phase="test", transform=data_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             shuffle=False, pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if args.resume is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 18, 25, 32], gamma=0.1)

    best_acc = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            _, alphas, pred = model(imgs)
            loss = criterion_cls(pred, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="Validation"):
                imgs, targets = imgs.to(device), targets.to(device)
                _, _, pred = model(imgs)
                loss = criterion_cls(pred, targets)
                val_loss += loss.item()
                _, predicts = torch.max(pred, 1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicts.cpu().numpy())

        val_acc = 100. * balanced_accuracy_score(y_true, y_pred)
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'./checkpoints/{args.exp_name}_best.pth')

        scheduler.step()

    print(f"Best Validation Accuracy: {best_acc:.2f}%")


    plot_metrics(train_losses, train_accs, val_losses, val_accs, args.exp_name)


if __name__ == "__main__":
    run_training()
