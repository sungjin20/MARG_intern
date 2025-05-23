import os
import numpy as np
from collections import defaultdict

# 설정
SEQ_LEN = 6
DATA_DIR = "/data2/personal/sungjin/research2/npy_data"
SPLIT_MONTH = "2016_01"  # 기준

# (1) 구 리스트 가져오기
files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npy")]
gu_month_dict = defaultdict(list)

for f in files:
    gu, ym = f.replace(".npy", "").split("_", 1)
    gu_month_dict[gu].append(ym)

# (2) 시계열 데이터 구성
X_train, Y_train, X_test, Y_test = [], [], [], []

for gu, month_list in gu_month_dict.items():
    month_list = sorted(month_list)  # 시계열 정렬

    images = []
    for ym in month_list:
        arr = np.load(os.path.join(DATA_DIR, f"{gu}_{ym}.npy"))
        images.append(arr)

    for i in range(len(images) - SEQ_LEN):
        input_seq = images[i:i+SEQ_LEN]      # (6, H, W)
        next_image = images[i+SEQ_LEN]       # 다음 달 이미지
        last_input_image = images[i+SEQ_LEN-1]  # 입력 시퀀스 마지막 이미지

        # 레이블을 다음 달 이미지와 시퀀스 마지막 이미지 차이의 절대값으로 설정
        label = np.abs(next_image - last_input_image)

        label_month = month_list[i+SEQ_LEN]

        if label_month < SPLIT_MONTH:
            X_train.append(np.stack(input_seq))
            Y_train.append(label)
        else:
            X_test.append(np.stack(input_seq))
            Y_test.append(label)


# (3) 배열로 저장
X_train = np.array(X_train, dtype=np.uint8)
Y_train = np.array(Y_train, dtype=np.uint8)
X_test = np.array(X_test, dtype=np.uint8)
Y_test = np.array(Y_test, dtype=np.uint8)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
np.save("/data2/personal/sungjin/research2/X_train_2.npy", X_train)
np.save("/data2/personal/sungjin/research2/Y_train_2.npy", Y_train)
np.save("/data2/personal/sungjin/research2/X_test_2.npy", X_test)
np.save("/data2/personal/sungjin/research2/Y_test_2.npy", Y_test)

import matplotlib.pyplot as plt

# (4) 샘플 하나 확인 (예: 첫 번째 train 데이터)
sample_idx = -1

x_sample = X_train[sample_idx]  # (SEQ_LEN, H, W)
y_sample = Y_train[sample_idx]  # (H, W)


# 입력 시퀀스 시각화
fig, axes = plt.subplots(1, SEQ_LEN + 1, figsize=(15, 3))

for i in range(SEQ_LEN):
    axes[i].imshow(x_sample[i], cmap='gray')
    axes[i].set_title(f"X[{i}]")
    axes[i].axis("off")

# 정답 이미지 시각화
axes[SEQ_LEN].imshow(y_sample, cmap='gray')
axes[SEQ_LEN].set_title("Y (label)")
axes[SEQ_LEN].axis("off")

plt.tight_layout()
plt.show()
