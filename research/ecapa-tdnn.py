import os
import torchaudio
import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

device = "cuda:11"

# 사전 훈련된 ECAPA-TDNN 모델 로드
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp",
    run_opts={"device": device}
)

# wav 폴더 경로
wav_folder = "/data2/personal/sungjin/korean_dialects/modified/Gyeongsang/wav"
embedding_folder = "/data2/personal/sungjin/korean_dialects/modified/Gyeongsang/embedding"

# wav 폴더 내의 모든 .wav 파일 목록 가져오기
wav_files = []
for root, dirs, files in os.walk(wav_folder):
    for file in files:
        if file.endswith(".wav"):
            wav_files.append(os.path.join(root, file))

wav_files.sort()
print(len(wav_files))
wav_files = wav_files[750000:]

# tqdm을 이용한 진행 상태 표시
for wav_path in tqdm(wav_files, desc="Embedding WAV files", unit="file"):
    try:
        # 하위 폴더 구조를 유지하며 임베딩 파일 경로 설정
        relative_path = os.path.relpath(os.path.dirname(wav_path), wav_folder)
        embedding_subfolder = os.path.join(embedding_folder, relative_path)
        embedding_file_path = os.path.join(embedding_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

        # 이미 임베딩 파일이 존재하면 건너뛰기
        if os.path.exists(embedding_file_path):
            continue

        # 음성 파일 로드
        signal, sr = torchaudio.load(wav_path)

        # GPU로 이동
        signal = signal.to(device)

        # 스피커 임베딩 추출
        with torch.no_grad():  # 메모리 절약
            embedding = model.encode_batch(signal)

        # GPU → CPU 변환 후 numpy 저장
        emb_vector = embedding.squeeze().detach().cpu().numpy()

        # 임베딩 저장할 폴더 생성
        os.makedirs(embedding_subfolder, exist_ok=True)

        # 임베딩 파일 저장
        np.save(embedding_file_path, emb_vector)

    except Exception as e:
        print(f"오류 발생: {e}. 파일 삭제: {wav_path}")
        try:
            # 오류 발생 시 파일 삭제
            os.remove(wav_path)
            print(f"파일 삭제됨: {wav_path}")
        except Exception as delete_error:
            print(f"파일 삭제 실패: {delete_error}")
        continue  # 오류 발생 시 다음 파일로 넘어감