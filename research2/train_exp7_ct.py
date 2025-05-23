import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# === 설정 ===
exp = 'exp_7'
resume_epoch = 644  # 이어서 학습할 시작 에폭 (예: 이전 학습이 1~40까지였다면 41부터 시작)
total_epochs = 2000  # 추가 학습 종료 시점 (예: 총 60 에폭까지 학습하고 싶을 경우)

print("🚀 TensorBoard 로그 디렉토리 생성 중...")
log_dir = os.path.join(f"/data2/personal/sungjin/research2/{exp}/runs", sorted(os.listdir(f"/data2/personal/sungjin/research2/{exp}/runs"))[-1])
writer = SummaryWriter(log_dir)
print(f"📁 기존 로그 디렉토리로 기록: {log_dir}")

# 하이퍼파라미터
SEQ_LEN = 6
IMG_SIZE = 256
BATCH_SIZE = 16
LR = 1e-3
DEVICE = 'cuda:4'
print(f"💻 실행 디바이스: {DEVICE}")

# 데이터 로드
X_train = np.load("/data2/personal/sungjin/research2/X_train_2.npy")
Y_train = np.load("/data2/personal/sungjin/research2/Y_train_2.npy")
X_test = np.load("/data2/personal/sungjin/research2/X_test_2.npy")
Y_test = np.load("/data2/personal/sungjin/research2/Y_test_2.npy")

X_train = X_train[:, :, np.newaxis, :, :]
X_test = X_test[:, :, np.newaxis, :, :]
Y_train = Y_train[:, np.newaxis, :, :]
Y_test = Y_test[:, np.newaxis, :, :]

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# === 모델 정의 ===
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTMNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)
        self.conv_out = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, W).to(x.device)
        c = torch.zeros(B, self.hidden_dim, H, W).to(x.device)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return self.conv_out(h)

# === 모델 불러오기 및 학습 환경 설정 ===
model = ConvLSTMNet().to(DEVICE)
checkpoint_path = f"/data2/personal/sungjin/research2/{exp}/conv_lstm_epoch_{resume_epoch:02d}.pt"
model.load_state_dict(torch.load(checkpoint_path))
print(f"✅ 모델 {resume_epoch} 에폭부터 이어서 학습 시작")

num_pos = np.sum(Y_train == 1)
num_neg = np.sum(Y_train == 0)
pos_weight_val = num_neg / (num_pos + 1e-6)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).cpu().numpy().astype(np.uint8)
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8)
    pred_flat = pred_bin.flatten()
    target_flat = target_bin.flatten()
    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_flat, zero_division=1)
    return precision, recall, f1

# === 학습 루프 ===
for epoch in range(resume_epoch, total_epochs):
    print(f"\n📘 Epoch {epoch+1}/{total_epochs}")
    
    model.train()
    train_loss, train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0, 0.0
    train_bar = tqdm(train_loader, desc="🛠️  Training", leave=False)

    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        p, r, f1 = compute_metrics(output, y_batch)
        train_precision += p
        train_recall += r
        train_f1 += f1
        train_bar.set_postfix(loss=loss.item(), precision=p, recall=r, f1=f1)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_precision = train_precision / len(train_loader)
    avg_train_recall = train_recall / len(train_loader)
    avg_train_f1 = train_f1 / len(train_loader)

    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Precision/Train", avg_train_precision, epoch)
    writer.add_scalar("Recall/Train", avg_train_recall, epoch)
    writer.add_scalar("F1/Train", avg_train_f1, epoch)

    model.eval()
    val_loss, val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0, 0.0
    val_bar = tqdm(test_loader, desc="🔍 Validation", leave=False)

    with torch.no_grad():
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
            p, r, f1 = compute_metrics(output, y_batch)
            val_precision += p
            val_recall += r
            val_f1 += f1
            val_bar.set_postfix(loss=loss.item(), precision=p, recall=r, f1=f1)

    avg_val_loss = val_loss / len(test_loader)
    avg_val_precision = val_precision / len(test_loader)
    avg_val_recall = val_recall / len(test_loader)
    avg_val_f1 = val_f1 / len(test_loader)

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Precision/Validation", avg_val_precision, epoch)
    writer.add_scalar("Recall/Validation", avg_val_recall, epoch)
    writer.add_scalar("F1/Validation", avg_val_f1, epoch)

    print(f"✅ Epoch {epoch+1}/{total_epochs} 완료 - "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train P/R/F1: {avg_train_precision:.4f}/{avg_train_recall:.4f}/{avg_train_f1:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val P/R/F1: {avg_val_precision:.4f}/{avg_val_recall:.4f}/{avg_val_f1:.4f}")

    torch.save(model.state_dict(), f"/data2/personal/sungjin/research2/{exp}/conv_lstm_epoch_{epoch+1:02d}.pt")

writer.close()
print("📉 TensorBoard 로그 기록 완료.")
