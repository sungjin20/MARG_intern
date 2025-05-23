import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import precision_score, recall_score, f1_score

# 💻 하이퍼파라미터
BATCH_SIZE = 4
DEVICE = 'cuda:1'
SEED = 50
NUM_SAMPLES = 100

# 시드 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 🧱 ConvLSTMCell, ConvLSTMNet 정의
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

# 데이터 로드 및 전처리
X_test = np.load("/data2/personal/sungjin/research2/X_test_2.npy")
Y_test = np.load("/data2/personal/sungjin/research2/Y_test_2.npy")

X_test = X_test[:, :, np.newaxis, :, :]
Y_test = Y_test[:, np.newaxis, :, :]

# 무작위 샘플 선택
total_samples = X_test.shape[0]
sample_indices = random.sample(range(total_samples), NUM_SAMPLES)
X_sample = X_test[sample_indices]
Y_sample = Y_test[sample_indices]

# TensorDataset & DataLoader
sample_dataset = TensorDataset(torch.tensor(X_sample, dtype=torch.float32),
                               torch.tensor(Y_sample, dtype=torch.float32))
sample_loader = DataLoader(sample_dataset, batch_size=1)

# 모델 로딩
model = ConvLSTMNet().to(DEVICE)
model.load_state_dict(torch.load("/data2/personal/sungjin/research2/exp_7/conv_lstm_epoch_15.pt", map_location=DEVICE))
model.eval()

def compute_metrics(pred, target, threshold=0.9834):
    pred_bin = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8)

    # flatten
    pred_flat = pred_bin.flatten()
    target_flat = target_bin.flatten()

    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_flat, zero_division=1)

    return precision, recall, f1

# 예측 시각화 함수
import matplotlib.patches as patches

def visualize_predictions(model, dataloader, device, save_dir="/data2/personal/sungjin/research2/exp_7/results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    shown = 0
    results = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = torch.sigmoid(model(x_batch))

            x_np = x_batch.cpu().numpy()
            y_np = y_batch.cpu().numpy()
            p_np = preds.cpu().numpy()

            for i in range(x_np.shape[0]):
                input_seq = x_np[i, :, 0]
                target_img = y_np[i, 0]
                pred_img = p_np[i, 0]
                pred_bin = (pred_img > 0.9834).astype(np.uint8)

                precision, recall, f1 = compute_metrics(torch.tensor(pred_img), torch.tensor(target_img))

                # 4개 subplot 생성
                fig, axes = plt.subplots(1, 4, figsize=(16, 6))
                
                # 1. 마지막 입력 프레임
                axes[0].imshow(input_seq[-1], cmap='gray')
                axes[0].set_title("Input (Last Frame)", fontsize=20)
                axes[0].axis('off')
                
                # 2. Ground Truth
                axes[1].imshow(target_img, cmap='gray')
                axes[1].set_title("Ground Truth", fontsize=20)
                axes[1].axis('off')
                
                # 3. 이진화된 Prediction
                axes[2].imshow(pred_bin, cmap='gray')
                axes[2].set_title(f"Prediction (F1: {f1:.4f})", fontsize=20)
                axes[2].axis('off')

                # 4. Prediction 빨간 점 표시, 배경 투명
                # 빈 캔버스에 투명 배경
                # 4번째 subplot
                axes[3].imshow(np.zeros_like(target_img), cmap='gray', alpha=0)  # 투명 배경
                ys, xs = np.where(target_img == 1)
                axes[3].scatter(xs, ys, color='red', s=5)
                axes[3].set_title("Ground Truth on Map", fontsize=20)

                # 축 눈금 및 라벨 없애기 (spines는 남김)
                axes[3].set_xticks([])
                axes[3].set_yticks([])

                # 경계선 보이게 설정
                for spine in axes[3].spines.values():
                    spine.set_visible(True)
                    spine.set_color('black')   # 검은색 테두리
                    spine.set_linewidth(1.5)



                # 배경 투명 설정은 저장할 때 figure에 지정
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"prediction_{shown}_seed{SEED}.png")
                plt.savefig(save_path, transparent=True)  # 투명 배경 저장 옵션
                plt.close(fig)

                results.append((save_path, f1))
                print(f"✅ Saved: {save_path} (Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f})")
                shown += 1

    print("\n📊 Top 5 predictions by F1 score:")
    top_results = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    for path, f1 in top_results:
        print(f"{path} (F1: {f1:.4f})")


# 🔍 실행
visualize_predictions(model, sample_loader, DEVICE)
