import torch
import torchaudio
import os
import torch.nn.functional as F
import numpy as np
from conformer import CustomConformer
import torchaudio.transforms as T

# 설정
region = 'Jeju'
test_wav_dir = f'/data2/personal/sungjin/korean_dialects/modified/{region}/wav'
trainset_file = f'/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_classification2_testset.txt'

# 테스트할 WAV 파일 목록 불러오기
with open(trainset_file, 'r') as f:
    test_files = [os.path.join(test_wav_dir, line.strip()) for line in f.readlines() if line.strip().endswith(".wav")]

# 디바이스 설정
device = torch.device('cuda:4')

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

# 반복적으로 체크포인트 변경하면서 평가
for epoch in range(2, 75, 4):  # 2, 4, 6, ..., 74 에폭 반복
    checkpoint_path = f'/data2/personal/sungjin/korean_dialects/classification_second_checkpoint/checkpoint_epoch_{epoch}.pth'
    
    # 모델 로드
    num_classes = 6
    model = CustomConformer(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n🔹 체크포인트: Epoch {epoch} 시작!")

    standard_count = 0
    for wav_path in test_files:
        wav_filename = os.path.basename(wav_path)

        hubert_path = wav_path.replace("/wav/", "/hubert/").replace(".wav", ".npy")
        f0_path = wav_path.replace("/wav/", "/f0/").replace(".wav", ".npy")

        hubert = torch.from_numpy(np.load(hubert_path)).T.unsqueeze(0)
        f0, uv = np.load(f0_path, allow_pickle=True)
        f0 = torch.FloatTensor(np.array(f0 * uv, dtype=float))
        f0 = F.interpolate(f0[None, None, :], size=hubert.shape[-1], mode='nearest').squeeze()
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        lf0 = lf0.T.unsqueeze(0)

        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 스테레오 → 모노 변환
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(device)
        hubert = hubert.to(device)
        lf0 = lf0.to(device)

        # Log-Mel 변환
        logmel = logmel_transform(waveform).squeeze(0)  
        logmel_energy = logmel.mean(dim=0).unsqueeze(0).unsqueeze(0)
        logmel_energy = F.interpolate(logmel_energy, size=hubert.shape[-1], mode='linear')

        inputs = torch.cat([logmel_energy, hubert, lf0], dim=1).permute(0, 2, 1)

        with torch.no_grad():
            output = model(inputs)
            probabilities = F.softmax(output, dim=1)
            probabilities_list = [round(p, 2) for p in probabilities.squeeze(0).tolist()]
            predicted_label = torch.argmax(probabilities, dim=1).item()

        if DIALECT_LABELS[predicted_label] == region:
            standard_count += 1

        print(f"파일: {wav_filename}, 예측된 방언: {DIALECT_LABELS[predicted_label]}, 확률: {probabilities_list}")

    # 정답 비율 출력
    standard_ratio = standard_count / len(test_files) * 100
    print(f"✅ Epoch {epoch}: 정답 비율: {standard_ratio:.2f}%")
