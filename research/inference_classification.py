import torch
import torchaudio
import os
import torch.nn.functional as F
from conformer import CustomConformer
import torchaudio.transforms as T
import numpy as np

# 체크포인트 경로 설정
checkpoint_path = '/data2/personal/sungjin/korean_dialects/checkpoint_real/checkpoint_epoch_36.pth'  # 최신 체크포인트 경로 설정

test_wav_dir = '/data2/personal/sungjin/korean_dialects/testset/Chungcheong/wav'
test_files = []
for root, _, files in os.walk(test_wav_dir):  # os.walk() 사용
    for f in files:
        if f.endswith('.wav'):
            test_files.append(os.path.join(root, f))  # 전체 경로 저장

# 디바이스 설정
device = torch.device('cuda:3')

# 모델 로드
num_classes = 6  # 학습 시 설정한 클래스 수와 동일해야 함
model = CustomConformer(num_classes=num_classes).to(device)

# 체크포인트 불러오기
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 모델을 평가 모드로 설정
model.eval()

# Log-Mel 변환기 정의
class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
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
        wav = F.pad(wav, ((self.n_fft - self.hop_length) // 2, (self.n_fft - self.hop_length) // 2), "reflect")
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

# 테스트 파일 목록 불러오기
with open(test_txt_path, 'r') as file:
    test_files = [line.strip() for line in file.readlines()]

# 추론 수행
results = []
for wav_path in test_files:
    wav_filename = os.path.basename(wav_path)  # 파일명만 추출

    # 오디오 로드
    waveform, sample_rate = torchaudio.load(wav_path)

    # 스테레오 → 모노 변환
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = waveform.to(device)

    # Log-Mel 변환
    logmel = logmel_transform(waveform).squeeze(0)  # [1, 80, T] → [80, T]
    logmel = logmel.unsqueeze(0).to(device)  # [1, 80, T]

    # 모델 예측
    with torch.no_grad():
        output = model(logmel)
        probabilities = F.softmax(output, dim=1)  # 확률값 변환
        predicted_label = torch.argmax(probabilities, dim=1).item()
        rounded_probs = np.round(probabilities.cpu().numpy(), 2)

    # 결과 저장
    results.append((wav_filename, DIALECT_LABELS[predicted_label], rounded_probs))

    # 결과 출력
    print(f"파일: {wav_filename}, 예측된 방언: {DIALECT_LABELS[predicted_label]}, 확률 분포: {rounded_probs}")