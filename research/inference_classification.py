import torch
import torchaudio
import os
import torch.nn.functional as F
import random
from conformer import CustomConformer
import torchaudio.transforms as T
import numpy as np

# 체크포인트 경로 설정
checkpoint_path = '/data2/personal/sungjin/korean_dialects/checkpoint_real/checkpoint_epoch_120.pth'

region = 'Gyeongsang'
test_wav_dir = '/data2/personal/sungjin/korean_dialects/classification_testset/' + region + '/wav'
test_files = []
for root, _, files in os.walk(test_wav_dir):
    for f in files:
        if f.endswith('.wav'):
            test_files.append(os.path.join(root, f))

random.shuffle(test_files)  # 파일 리스트를 랜덤하게 섞음
test_files = test_files[:100]  # 랜덤하게 100개 선택

# 디바이스 설정
device = torch.device('cuda:6')

# 모델 로드
num_classes = 6
model = CustomConformer(num_classes=num_classes).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Log-Mel 변환기 정의
class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False):
        super().__init__()
        self.melspctrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            power=1.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="slaney",
        )

    def forward(self, wav):
        mel = self.melspctrogram(wav)
        logmel = torch.log(torch.clamp(mel, min=1e-5))
        return logmel

logmel_transform = LogMelSpectrogram().to(device)

# 레이블 매핑
DIALECT_LABELS = {
    0: 'standard',
    1: 'Chungcheong',
    2: 'Gangwon',
    3: 'Gyeongsang',
    4: 'Jeju',
    5: 'Jeolla'
}

# 추론 수행
standard_count = 0
for wav_path in test_files:
    wav_filename = os.path.basename(wav_path)
    waveform, sample_rate = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)
    logmel = logmel_transform(waveform).squeeze(0)
    logmel = logmel.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(logmel)
        probabilities = F.softmax(output, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    if DIALECT_LABELS[predicted_label] == region:
        standard_count += 1

    print(f"파일: {wav_filename}, 예측된 방언: {DIALECT_LABELS[predicted_label]}")

# standard 비율 출력
standard_ratio = standard_count / len(test_files) * 100
print(f"정답 비율: {standard_ratio:.2f}%")
