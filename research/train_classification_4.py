from conformer import CustomConformer3
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
import numpy as np

# 디바이스 설정
device = torch.device('cuda:4')

# 하이퍼파라미터 설정
num_epochs = 1000
batch_size = 16
num_classes = 6
save_checkpoint_epoch = 2
checkpoint_save_dir = '/data2/personal/sungjin/korean_dialects/classification_fourth_checkpoint'
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

# 커스텀 데이터셋 클래스
class CustomDialectDataset(Dataset):
    def __init__(self, dialect_data_dir, standard_data_dir, regions, mode='train', transform=None):
        self.data = []
        self.transform = transform
        
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
        f0_path = wav_path.replace("/wav/", "/f0/").replace(".wav", ".npy")

        f0, uv = np.load(f0_path, allow_pickle=True)
        f0 = torch.FloatTensor(np.array(f0*uv,dtype=float))
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        lf0 = lf0.T

        return lf0, label

# Collate Function 정의
def collate_fn(batch):
    lf0s, labels = zip(*batch)

    lengths = torch.tensor([lf0.shape[1] for lf0 in lf0s], dtype=torch.long)
    max_len = max(lengths)

    padded_lf0s = torch.stack([
        F.pad(lf0, (0, max_len - lf0.shape[1]), "constant", 0) for lf0 in lf0s
    ]).permute(0, 2, 1)

    # Label 텐서 변환
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_lf0s, labels


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
model = CustomConformer3(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

# TensorBoard 설정
writer = SummaryWriter(log_dir='/data2/personal/sungjin/korean_dialects/classification_tensorboard_logs_fourth')

# 로그 설정
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정
    format='%(asctime)s - %(message)s',  # 로그 포맷 설정
    handlers=[logging.FileHandler('classification_train_log_fourth.txt'), logging.StreamHandler()]  # 로그를 파일과 콘솔로 출력
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
