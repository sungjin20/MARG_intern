import os
import torchaudio
import torch
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

device = "cuda:12"

# 사전 훈련된 ECAPA-TDNN 모델 로드
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp",
    run_opts={"device": device}
)

# 특정 파일 경로 설정
specific_wav_path = "/data2/personal/sungjin/korean_dialects/modified/Gyeongsang/wav/DKSR20004045/DKSR20004045_1_1_58.wav"
embedding_folder = "/data2/personal/sungjin/korean_dialects/modified/Gyeongsang/embedding"
embedding_file_path = "/data2/personal/sungjin/korean_dialects/modified/Gyeongsang/embedding/DKSR20004045/DKSR20004045_1_1_58.npy"

# 파일 경로가 존재하는지 확인
if os.path.exists(specific_wav_path):
    try:
        # 임베딩 파일이 이미 존재하면 건너뛰기
        if os.path.exists(embedding_file_path):
            print(f"임베딩 파일이 이미 존재합니다: {embedding_file_path}")
        else:
            # 음성 파일 로드
            signal, sr = torchaudio.load(specific_wav_path)

            # 스테레오인 경우 모노로 변환 (채널 수가 2일 때)
            if signal.shape[0] == 2:
                signal = signal.mean(dim=0, keepdim=True)

            # GPU로 이동
            signal = signal.to(device)

            # 스피커 임베딩 추출
            with torch.no_grad():  # 메모리 절약
                embedding = model.encode_batch(signal)

            # GPU → CPU 변환 후 numpy 저장
            emb_vector = embedding.squeeze().detach().cpu().numpy()

            # 임베딩 파일 저장
            os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)
            np.save(embedding_file_path, emb_vector)
            print(f"임베딩 파일이 생성되었습니다: {embedding_file_path}")

    except Exception as e:
        print(f"오류 발생: {e}")
else:
    print(f"지정된 WAV 파일이 존재하지 않습니다: {specific_wav_path}")
