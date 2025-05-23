import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

# 환경 설정
DEVICE = 'cuda:1'
exp = 'exp_7'
CHECKPOINT_EPOCH = 15

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
        self.conv_out = nn.Conv2d(hidden_dims[-1], 1, kernel_size=1)

    def forward(self, x):  # (B, T, C, H, W)
        B, T, C, H, W = x.size()
        device = x.device
        h = [torch.zeros(B, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dims[i], H, W, device=device) for i in range(self.num_layers)]

        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.layers):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]

        out = self.conv_out(h[-1])
        return out

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

#model = MultiLayerConvLSTM(input_dim=1, hidden_dims=[32, 64], kernel_size=3).to(DEVICE)
model = ConvLSTMNet().to(DEVICE)
checkpoint_path = f"/data2/personal/sungjin/research2/{exp}/conv_lstm_epoch_{CHECKPOINT_EPOCH:02d}.pt"
print(f"📦 체크포인트 로드 중: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
print("✅ 체크포인트 로드 완료!")

# 데이터 로드 (차원 추가 및 TensorDataset 생성 포함)
X_test = np.load("/data2/personal/sungjin/research2/X_test_2.npy")
Y_test = np.load("/data2/personal/sungjin/research2/Y_test_2.npy")
X_test = X_test[:, :, np.newaxis, :, :]
Y_test = Y_test[:, np.newaxis, :, :]
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                              torch.tensor(Y_test, dtype=torch.float32))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

# 전체 예측 수집
print("🔍 Validation 세트에서 Threshold 튜닝 중...")
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in tqdm(test_loader, desc="📊 Collecting predictions"):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        output = model(x_batch)
        probs = torch.sigmoid(output).cpu().numpy()
        targets = y_batch.cpu().numpy()
        all_preds.append(probs.flatten())
        all_targets.append(targets.flatten())

# numpy 배열로 변환
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

# PR 곡선 계산
precisions, recalls, thresholds = precision_recall_curve(all_targets, all_preds)

# F1 score 계산
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

# 결과 출력
print(f"\n🎯 최적 Threshold: {best_threshold:.4f}")
print(f"✅ Precision: {precisions[best_idx]:.4f}")
print(f"✅ Recall   : {recalls[best_idx]:.4f}")
print(f"✅ F1 Score : {f1_scores[best_idx]:.4f}")

# 그래프 출력
plt.figure(figsize=(10, 8))

# PR Curve
plt.plot(recalls, precisions, label='PR Curve', linewidth=2)

# 최적 threshold 표시
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Best Threshold = {best_threshold:.4f}', markersize=10)

# 폰트 크기 조정 부분
plt.title("Precision-Recall Curve", fontsize=30)
plt.xlabel("Recall", fontsize=25)
plt.ylabel("Precision", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)

plt.grid(True)
plt.tight_layout()
plt.savefig(f"/data2/personal/sungjin/research2/{exp}/pr_curve_epoch{CHECKPOINT_EPOCH}.png")
plt.close()

# Threshold vs Precision & Recall 그래프
plt.figure(figsize=(10, 8))
plt.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
#plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')

plt.title("Threshold vs Precision & Recall", fontsize=30)
plt.xlabel("Threshold", fontsize=25)
plt.ylabel("Score", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/data2/personal/sungjin/research2/{exp}/threshold_precision_recall_epoch{CHECKPOINT_EPOCH}.png")
plt.close()

# Threshold vs F1 Score 그래프
plt.figure(figsize=(10, 8))
plt.plot(thresholds, f1_scores[:-1], label='F1 Score', color='green', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best Threshold = {best_threshold:.4f}')

plt.title("Threshold vs F1 Score", fontsize=30)
plt.xlabel("Threshold", fontsize=25)
plt.ylabel("F1 Score", fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"/data2/personal/sungjin/research2/{exp}/threshold_f1score_epoch{CHECKPOINT_EPOCH}.png")
plt.close()
