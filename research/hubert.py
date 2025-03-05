import torch
import numpy as np
import os
import torchaudio
from tqdm import tqdm
from transformers import HubertModel

device = "cuda:4"

model = HubertModel.from_pretrained("team-lucid/hubert-base-korean").to(device)

# wav 파일들이 상대경로로 저장된 trainset 파일 경로
trainset_file = "/data2/personal/sungjin/korean_dialects/modified/Jeolla/Jeolla_classification2_testset.txt"

# wav 폴더 경로
wav_folder = "/data2/personal/sungjin/korean_dialects/modified/Jeolla/wav"

hubert_folder = "/data2/personal/sungjin/korean_dialects/modified/Jeolla/hubert"
os.makedirs(hubert_folder, exist_ok=True)

# trainset_file에서 .wav 파일 목록 읽기
wav_files = []
with open(trainset_file, 'r') as f:
    wav_files = [os.path.join(wav_folder, line.strip()) for line in f.readlines() if line.strip().endswith(".wav")]

wav_files.sort()
print(f"Number of wav files: {len(wav_files)}")

# tqdm을 이용한 진행 상태 표시
for wav_path in tqdm(wav_files, desc="Make hubert from WAV files", unit="file"):
    # 하위 폴더 구조를 유지하며 임베딩 파일 경로 설정
    relative_path = os.path.relpath(os.path.dirname(wav_path), wav_folder)
    hubert_subfolder = os.path.join(hubert_folder, relative_path)
    hubert_file_path = os.path.join(hubert_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

    # 이미 임베딩 파일이 존재하면 건너뛰기
    if os.path.exists(hubert_file_path):
        continue

    # 음성 파일 로드
    signal, sr = torchaudio.load(wav_path)

    # 스테레오인 경우 모노로 변환 (채널 수가 2일 때)
    if signal.shape[0] == 2:
        signal = signal.mean(dim=0, keepdim=True)

    signal = signal.to(device)  # GPU로 이동

    # 스피커 임베딩 추출
    with torch.no_grad():  # 메모리 절약
        hubert = model(signal)

    # GPU → CPU 변환 후 numpy 저장
    hubert_vector = hubert.last_hidden_state.squeeze().detach().cpu().numpy()

    # 임베딩 저장할 폴더 생성
    os.makedirs(hubert_subfolder, exist_ok=True)

    # 임베딩 파일 저장
    np.save(hubert_file_path, hubert_vector)
