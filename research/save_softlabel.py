import torch
import numpy as np
import os
import torchaudio
from tqdm import tqdm
from conformer import CustomConformer2
import torchaudio.transforms as T
import torch.nn.functional as F

device = "cuda:2"
epoch = 18
num_classes = 6
region = 'Jeolla'
mode = 'valid'
phoneme_dir = f'/data2/personal/sungjin/korean_standard/phoneme'
wav_folder = f"/data2/personal/sungjin/korean_dialects/modified/{region}/wav"
checkpoint_path = f'/data2/personal/sungjin/korean_dialects/classification_third_checkpoint/checkpoint_epoch_{epoch}.pth'
model = CustomConformer2(num_classes=num_classes).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# wav 파일들이 상대경로로 저장된 trainset 파일 경로
trainset_file = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_speech_{mode}set.txt"
test_files = []
wav_files = []
with open(trainset_file, 'r') as f:
    for line in f.readlines():
        if line.strip().endswith(".wav"):
            test_files.append(os.path.join(phoneme_dir, line.strip().replace(".wav", ".npy")))
            wav_files.append(os.path.join(wav_folder, line.strip()))

DIALECT_LABELS = {
    'standard': 0,
    'Chungcheong': 1,
    'Gangwon': 2,
    'Gyeongsang': 3,
    'Jeju': 4,
    'Jeolla': 5
}

softlabel_folder = f"/data2/personal/sungjin/korean_dialects/modified/{region}/softlabel"
os.makedirs(softlabel_folder, exist_ok=True)

# tqdm을 이용한 진행 상태 표시
for phoneme_path, wav_path in tqdm(zip(test_files, wav_files), desc="Make softlabel from phoneme files", unit="file"):
    # 하위 폴더 구조를 유지하며 임베딩 파일 경로 설정
    relative_path = os.path.relpath(os.path.dirname(wav_path), wav_folder)
    softlabel_subfolder = os.path.join(softlabel_folder, relative_path)
    softlabel_file_path = os.path.join(softlabel_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

    # 이미 임베딩 파일이 존재하면 건너뛰기
    if os.path.exists(softlabel_file_path):
        continue

    inputs = torch.from_numpy(np.load(phoneme_path)).unsqueeze(0)
    inputs = inputs.to(device)

    # 스피커 임베딩 추출
    with torch.no_grad():  # 메모리 절약
        output = model(inputs)
    output = output.detach().cpu()
    probabilities = F.softmax(output, dim=1)
    probabilities_list = [p for p in probabilities.squeeze(0).tolist()]

    standard_val = 0
    for i in range(len(probabilities_list)):
        if i != DIALECT_LABELS[region]:
            standard_val += probabilities_list[i]
            probabilities_list[i] = 0
    probabilities_list[0] = standard_val

    # 임베딩 저장할 폴더 생성
    os.makedirs(softlabel_subfolder, exist_ok=True)

    # 임베딩 파일 저장
    np.save(softlabel_file_path, np.array(probabilities_list))