from conformer import CustomConformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 추가
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 디바이스 설정
device = torch.device('cuda:3')

# 하이퍼파라미터 설정
num_epochs = 1000
batch_size = 32
num_classes = 6
save_checkpoint_epoch = 2
checkpoint_save_dir = '/data2/personal/sungjin/korean_dialects/checkpoint_real'
dialect_data_dir = '/data2/personal/sungjin/korean_dialects/modified'
standard_data_dir = '/data2/personal/sungjin/korean_standard'

# 레이블 매핑
DIALECT_LABELS = {
    'standard': 0,
    'Chungcheong': 1,
    'Gangwon': 2,
    'Gyeongsang': 3,
    'Jeju': 4,
    'Jeolla': 5
}

class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, center):
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


# 커스텀 데이터셋 클래스
class CustomDialectDataset(Dataset):
    def __init__(self, dialect_data_dir, standard_data_dir, regions, mode='train', transform=None):
        self.data = []
        self.transform = transform
        self.logmel = LogMelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, center=False)
        
        for region in regions:
            label = DIALECT_LABELS[region]

            # 표준어 데이터 경로 설정
            if region == 'standard':
                txt_file = os.path.join(standard_data_dir, f'standard_classification_{mode}set.txt')
                base_dir = os.path.join(standard_data_dir, 'wav')
            else:
                txt_file = os.path.join(dialect_data_dir, region, f"{region}_classification_{mode}set.txt")
                base_dir = os.path.join(dialect_data_dir, region, 'wav')
            
            # 텍스트 파일에서 wav 파일 경로 읽기
            with open(txt_file, 'r') as file:
                lines = file.readlines()
                
            for line in lines:
                wav_path = os.path.join(base_dir, line.strip())
                self.data.append((wav_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        wav_path, label = self.data[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        
        # 모든 오디오를 모노로 변환
        if waveform.shape[0] > 1:  # 스테레오 → 모노 변환
            waveform = waveform.mean(dim=0, keepdim=True)

        # Log-Mel 변환
        logmel = self.logmel(waveform).squeeze(0)  # [1, 80, T] → [80, T]

        return logmel, label

# Collate Function 정의
def collate_fn(batch):
    logmels, labels = zip(*batch)

    # 모든 샘플의 길이 측정
    lengths = torch.tensor([logmel.shape[1] for logmel in logmels], dtype=torch.long)

    # 가장 긴 샘플의 길이에 맞춰 패딩
    max_len = max(lengths)
    padded_logmel = torch.stack([
        F.pad(logmel, (0, max_len - logmel.shape[1]), "constant", 0) for logmel in logmels
    ])

    # 레이블을 텐서로 변환
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_logmel, labels

# 지역 설정
regions = ['standard', 'Chungcheong', 'Gangwon', 'Gyeongsang', 'Jeju', 'Jeolla']

# 데이터셋 생성
train_dataset = CustomDialectDataset(dialect_data_dir, standard_data_dir, regions, mode='train')
val_dataset = CustomDialectDataset(dialect_data_dir, standard_data_dir, regions, mode='valid')

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24, pin_memory=True, collate_fn=collate_fn)

## 모델 학습 코드 ##
os.makedirs(checkpoint_save_dir, exist_ok=True)
criterion = nn.CrossEntropyLoss().to(device)
model = CustomConformer(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

# TensorBoard 설정
writer = SummaryWriter(log_dir='/data2/personal/sungjin/korean_dialects/classification_tensorboard_logs_real')

# 로그 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정
    format='%(asctime)s - %(message)s',  # 로그 포맷 설정
    handlers=[logging.FileHandler('classification_train_log_real.txt'), logging.StreamHandler()]  # 로그를 파일과 콘솔로 출력
)

torch.cuda.empty_cache()
# 학습 코드 수정
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for step, (inputs, labels) in enumerate(train_loader, 1):  # step 카운트 시작을 1로 설정
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)  # FP32로 연산
        loss = criterion(outputs, labels)

        loss.backward()  # 일반적인 FP32 학습
        optimizer.step()
        
        total_loss += loss.item()

        # 100 step마다 진행 상황 출력
        if step % 100 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = total_loss / len(train_loader)

    # TensorBoard에 train loss 기록
    writer.add_scalar('Train/Loss', avg_train_loss, epoch)

    # 평가 단계 코드 수정
    model.eval()
    total_val_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for step, (inputs, labels) in enumerate(val_loader, 1):  # step 카운트 시작을 1로 설정
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            val_loss = criterion(outputs, labels)

            total_val_loss += val_loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, dim=1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # 100 step마다 진행 상황 출력
            if step % 100 == 0:
                logging.info(f'Epoch [{epoch+1}/{num_epochs}], Validation Step [{step}/{len(val_loader)}], Val Loss: {val_loss.item():.4f}')

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = (correct_predictions / total_samples) * 100

    # TensorBoard에 validation loss와 accuracy 기록
    writer.add_scalar('Validation/Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)

    # 결과 출력
    logging.info(f'Epoch {epoch+1}/{num_epochs}, '
                 f'Train Loss: {avg_train_loss:.4f}, '
                 f'Val Loss: {avg_val_loss:.4f}, '
                 f'Accuracy: {accuracy:.2f}%')

    # 학습률 스케줄러 적용 (검증 손실을 기준으로 학습률 감소)
    scheduler.step(avg_val_loss)

    ### 모델 저장 ###
    if (epoch + 1) % save_checkpoint_epoch == 0:
        model_path = os.path.join(checkpoint_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'accuracy': accuracy
        }, model_path)
        logging.info(f'Model saved at {model_path}')

# TensorBoard 종료
writer.close()
