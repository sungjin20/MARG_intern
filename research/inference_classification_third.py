import torch
import torchaudio
import os
import torch.nn.functional as F
import numpy as np
from conformer import CustomConformer2
import torchaudio.transforms as T

# 설정
region = 'Jeju'
phoneme_dir = f'/data2/personal/sungjin/korean_standard/phoneme'
trainset_file = f'/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_classification2_testset.txt'

# 테스트할 WAV 파일 목록 불러오기
with open(trainset_file, 'r') as f:
    test_files = [os.path.join(phoneme_dir, line.strip().replace(".wav", ".npy")) for line in f.readlines() if line.strip().endswith(".wav")]

# 디바이스 설정
device = torch.device('cuda:11')

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
for epoch in range(6, 20, 2):  # 2, 4, 6, ..., 74 에폭 반복
    checkpoint_path = f'/data2/personal/sungjin/korean_dialects/classification_third_checkpoint/checkpoint_epoch_{epoch}.pth'
    
    # 모델 로드
    num_classes = 6
    model = CustomConformer2(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\n🔹 체크포인트: Epoch {epoch} 시작!")

    standard_count = 0
    for phoneme_path in test_files:
        wav_path = phoneme_path.replace("korean_standard/phoneme/", f"korean_dialects/modified/{region}/wav/").replace(".npy", ".wav")
        inputs = torch.from_numpy(np.load(phoneme_path)).unsqueeze(0)
        inputs = inputs.to(device)

        with torch.no_grad():
            output = model(inputs)
            probabilities = F.softmax(output, dim=1)
            probabilities_list = [f"{p:.2f}" for p in probabilities.squeeze(0).tolist()]
            predicted_label = torch.argmax(probabilities, dim=1).item()

        if DIALECT_LABELS[predicted_label] == region:
            standard_count += 1

        print(f"확률: {probabilities_list}, 예측된 방언: {DIALECT_LABELS[predicted_label]}, 파일 : {wav_path}")

    # 정답 비율 출력
    standard_ratio = standard_count / len(test_files) * 100
    print(f"✅ Epoch {epoch}: 정답 비율: {standard_ratio:.2f}%")
