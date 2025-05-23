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

exp = 'exp_6'
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
DEVICE = 'cuda:4'
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

class MultiLayerConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=[32, 64], kernel_size=3):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layers.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)  # Output 1 channel (segmentation mask)

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.size()
        device = x.device

        # initialize hidden states and cell states for each layer
        h = [torch.zeros(B, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t]  # shape: (B, C, H, W)
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]  # output of this layer is input to next layer

        # use final hidden state from the last layer
        out = self.conv_out(h[-1])  # shape: (B, 1, H, W)
        return out

# 학습 시작
print("📦 모델 초기화 중...")
model = MultiLayerConvLSTM(input_dim=1, hidden_dims=[32, 64], kernel_size=3).to(DEVICE)

# 👇 pos_weight 자동 계산
num_pos = np.sum(Y_train == 1)
num_neg = np.sum(Y_train == 0)
pos_weight_val = num_neg / (num_pos + 1e-6)
print(num_pos)
print(num_neg)
print(f"📊 자동 계산된 pos_weight: {pos_weight_val:.2f}")
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)

# 손실 함수 정의
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred) > threshold).cpu().numpy().astype(np.uint8)
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8)

    # flatten
    pred_flat = pred_bin.flatten()
    target_flat = target_bin.flatten()

    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_flat, zero_division=1)

    return precision, recall, f1

print("🚀 학습 시작!")
for epoch in range(EPOCHS):
    print(f"\n📘 Epoch {epoch+1}/{EPOCHS}")
    
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc="🛠️  Training", leave=False)
    train_precision, train_recall, train_f1 = 0.0, 0.0, 0.0  # ⭐ 초기화

    for x_batch, y_batch in train_bar:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        p, r, f1 = compute_metrics(output, y_batch)  # ⭐ precision, recall, f1
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
    val_loss = 0.0
    val_bar = tqdm(test_loader, desc="🔍 Validation", leave=False)
    val_precision, val_recall, val_f1 = 0.0, 0.0, 0.0  # ⭐ 초기화

    with torch.no_grad():
        for x_batch, y_batch in val_bar:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()

            p, r, f1 = compute_metrics(output, y_batch)  # ⭐ precision, recall, f1
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

    print(f"✅ Epoch {epoch+1}/{EPOCHS} 완료 - "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train P/R/F1: {avg_train_precision:.4f}/{avg_train_recall:.4f}/{avg_train_f1:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val P/R/F1: {avg_val_precision:.4f}/{avg_val_recall:.4f}/{avg_val_f1:.4f}")



        # 에폭별 모델 저장
    torch.save(model.state_dict(), f"/data2/personal/sungjin/research2/{exp}/conv_lstm_epoch_{epoch+1:02d}.pt")

writer.close()
print("📉 TensorBoard 로그 기록 완료.")
