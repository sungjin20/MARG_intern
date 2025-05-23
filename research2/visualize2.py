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
SEED = 42
NUM_SAMPLES = 5

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
model = MultiLayerConvLSTM(input_dim=1, hidden_dims=[32, 64], kernel_size=3).to(DEVICE)
model.load_state_dict(torch.load("/data2/personal/sungjin/research2/exp_6/conv_lstm_epoch_536.pt", map_location=DEVICE))
model.eval()

def compute_metrics(pred, target, threshold=0.3):
    pred_bin = (pred > threshold).cpu().numpy().astype(np.uint8)
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8)

    pred_flat = pred_bin.flatten()
    target_flat = target_bin.flatten()

    precision = precision_score(target_flat, pred_flat, zero_division=1)
    recall = recall_score(target_flat, pred_flat, zero_division=1)
    f1 = f1_score(target_flat, pred_flat, zero_division=1)

    return precision, recall, f1

# 예측 시각화 함수
def visualize_predictions(model, dataloader, device, save_dir="/data2/personal/sungjin/research2/exp_6/results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    shown = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = torch.sigmoid(model(x_batch))

            x_np = x_batch.cpu().numpy()
            y_np = y_batch.cpu().numpy()
            p_np = preds.cpu().numpy()

            for i in range(x_np.shape[0]):
                input_seq = x_np[i, :, 0]      # (T, H, W)
                target_img = y_np[i, 0]        # (H, W)
                pred_img = p_np[i, 0]          # (H, W)
                pred_bin = (pred_img > 0.3).astype(np.uint8)

                error_img = np.abs(pred_bin - target_img)
                diff_img = np.abs(target_img - input_seq[-1])
                diff_img2 = np.abs(pred_bin - input_seq[-1])

                precision, recall, f1 = compute_metrics(torch.tensor(pred_img), torch.tensor(target_img))

                # 시각화
                fig, axes = plt.subplots(1, 6, figsize=(25, 4))
                axes[0].imshow(input_seq[-1], cmap='gray')
                axes[0].set_title("Input (Last Frame)")
                axes[1].imshow(target_img, cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred_bin, cmap='gray')
                axes[2].set_title("Prediction")
                axes[3].imshow(diff_img, cmap='hot')
                axes[3].set_title("Input - GT Diff")
                axes[4].imshow(diff_img2, cmap='hot')
                axes[4].set_title("Input - Pred Diff")
                axes[5].imshow(error_img, cmap='hot')
                axes[5].set_title(f"GT - Pred Diff\nP: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}")
                
                for ax in axes:
                    ax.axis('off')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f"prediction_{shown}_seed{SEED}.png")
                plt.savefig(save_path)
                plt.close(fig)

                print(f"✅ Saved: {save_path} (P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f})")
                shown += 1


# 🔍 실행
visualize_predictions(model, sample_loader, DEVICE)
