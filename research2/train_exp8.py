import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import os

exp = 'exp_8'
print("🚀 TensorBoard 로그 디렉토리 생성 중...")
log_dir = os.path.join(f"/data2/personal/sungjin/research2/{exp}/runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
writer = SummaryWriter(log_dir)
print(f"📁 로그 저장 위치: {log_dir}")

# 하이퍼파라미터
SEQ_LEN = 6
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
DEVICE = 'cuda:8'
print(f"💻 실행 디바이스: {DEVICE}")

# 데이터 로드
print("📂 NumPy 데이터 로딩 중...")
X_train = np.load("/data2/personal/sungjin/research2/X_train_2.npy")  # (N, 12, 512, 512)
Y_train = np.load("/data2/personal/sungjin/research2/Y_train_2.npy")
X_test = np.load("/data2/personal/sungjin/research2/X_test_2.npy")
Y_test = np.load("/data2/personal/sungjin/research2/Y_test_2.npy")

print("✅ NumPy 데이터 로드 완료!")

# 차원 추가
print("🔄 채널 차원 추가 중...")
X_train = X_train[:, :, np.newaxis, :, :]
X_test = X_test[:, :, np.newaxis, :, :]
Y_train = Y_train[:, np.newaxis, :, :]
Y_test = Y_test[:, np.newaxis, :, :]

# Tensor 변환
print("🧱 TensorDataset 생성 중...")
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(Y_test, dtype=torch.float32))

print("🔄 DataLoader 생성 중...")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 모델 정의
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

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        h = torch.zeros(B, self.hidden_dim, H, W).to(x.device)
        c = torch.zeros(B, self.hidden_dim, H, W).to(x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        return self.conv_out(h)

# 학습 시작
print("📦 모델 초기화 중...")
model = ConvLSTMNet().to(DEVICE)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)  # Ensure sigmoid activation
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice

def bce_dice_loss(pred, target, alpha=0.1):
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return alpha * bce + (1 - alpha) * dice

pos_weight_val = 1000
print(f"📊 pos_weight: {pos_weight_val:.2f}")
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)

# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=LR)

def accuracy_metric(pred, target, threshold=0.5):
    """
    target == 1인 부분에서 pred도 1로 맞춘 비율 (Recall-like)
    """
    pred_bin = torch.sigmoid(pred) > threshold
    target_bin = target > 0.5

    # target이 1인 위치에서 pred도 1인지 확인
    true_positive = (pred_bin & target_bin).float().sum()
    total_positive = target_bin.float().sum()

    if total_positive == 0:
        return torch.tensor(1.0)  # target에 1이 없으면 정확히 예측했다고 간주

    return true_positive / total_positive

print("🚀 학습 시작!")
for epoch in range(EPOCHS):
    print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")
    
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_bar = tqdm(train_loader, desc="🛠️  Training", leave=False)
    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        acc = accuracy_metric(output, y_batch)
        train_acc += acc.item()

        train_bar.set_postfix(loss=loss.item(), acc=acc.item())

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)
    writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_bar = tqdm(test_loader, desc="🔍 Validation", leave=False)
    with torch.no_grad():
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

            acc = accuracy_metric(output, y_batch)
            val_acc += acc.item()

            val_bar.set_postfix(loss=loss.item(), acc=acc.item())

    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = val_acc / len(test_loader)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

    print(f"✅ Epoch {epoch+1}/{EPOCHS} 완료 - "f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")


        # 에폭별 모델 저장
    torch.save(model.state_dict(), f"/data2/personal/sungjin/research2/{exp}/conv_lstm_epoch_{epoch+1:02d}.pt")

writer.close()
print("📉 TensorBoard 로그 기록 완료.")
