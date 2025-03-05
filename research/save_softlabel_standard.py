import numpy as np
import os
from tqdm import tqdm

region = 'standard'
mode = 'valid'
wav_folder = f"/data2/personal/sungjin/korean_standard/wav"

# wav 파일들이 상대경로로 저장된 trainset 파일 경로
trainset_file = f"/data2/personal/sungjin/korean_standard/{region}_speech_{mode}set.txt"
wav_files = []
with open(trainset_file, 'r') as f:
    for line in f.readlines():
        if line.strip().endswith(".wav"):
            wav_files.append(os.path.join(wav_folder, line.strip()))

softlabel_folder = f"/data2/personal/sungjin/korean_standard/softlabel"
os.makedirs(softlabel_folder, exist_ok=True)

# tqdm을 이용한 진행 상태 표시
for wav_path in tqdm(wav_files, desc="Make softlabel from phoneme files", unit="file"):
    # 하위 폴더 구조를 유지하며 임베딩 파일 경로 설정
    relative_path = os.path.relpath(os.path.dirname(wav_path), wav_folder)
    softlabel_subfolder = os.path.join(softlabel_folder, relative_path)
    softlabel_file_path = os.path.join(softlabel_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

    # 이미 임베딩 파일이 존재하면 건너뛰기
    if os.path.exists(softlabel_file_path):
        continue

    probabilities_list = [1, 0, 0, 0, 0, 0]

    # 임베딩 저장할 폴더 생성
    os.makedirs(softlabel_subfolder, exist_ok=True)

    # 임베딩 파일 저장
    np.save(softlabel_file_path, np.array(probabilities_list))