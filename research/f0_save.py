import os
import librosa
import numpy as np
from tqdm import tqdm
from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
import torch

device = "cuda:4"
f0_predictor = FCPEF0Predictor(hop_length=512,sampling_rate=44100,dtype=torch.float32 ,device=device,threshold=0.05)

trainset_file = "/data2/personal/sungjin/korean_dialects/modified/Jeju/Jeju_classification2_testset.txt"
wav_folder = "/data2/personal/sungjin/korean_dialects/modified/Jeju/wav"
f0_folder = "/data2/personal/sungjin/korean_dialects/modified/Jeju/f0"
os.makedirs(f0_folder, exist_ok=True)

# trainset_file에서 .wav 파일 목록 읽기
wav_files = []
with open(trainset_file, 'r') as f:
    wav_files = [os.path.join(wav_folder, line.strip()) for line in f.readlines() if line.strip().endswith(".wav")]

wav_files.sort()
print(f"Number of wav files: {len(wav_files)}")

for wav_path in tqdm(wav_files, desc="Make f0 from WAV files", unit="file"):
    relative_path = os.path.relpath(os.path.dirname(wav_path), wav_folder)
    f0_subfolder = os.path.join(f0_folder, relative_path)
    f0_file_path = os.path.join(f0_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

    # 이미 임베딩 파일이 존재하면 건너뛰기
    if os.path.exists(f0_file_path):
        continue

    wav16, sr = librosa.load(wav_path, sr=16000)
    wav44 = librosa.resample(wav16, orig_sr=16000, target_sr=44100)
    f0_441, uv = f0_predictor.compute_f0_uv(wav44)
    os.makedirs(f0_subfolder, exist_ok=True)
    np.save(f0_file_path, np.array((f0_441, uv), dtype=object))