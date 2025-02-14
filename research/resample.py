import librosa
import soundfile as sf
import os
from tqdm import tqdm  # 진행 상태 표시

def resample_wav_folder_recursive(folder_path, target_sr=16000):
    # 모든 하위 폴더의 .wav 파일 찾기
    wav_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    for file_path in tqdm(wav_files, desc="Resampling WAV files"):
        try:
            # 오디오 로드 (원본 샘플링 레이트 유지)
            y, sr = librosa.load(file_path, sr=None)
            
            # 원본 샘플링 레이트가 목표 샘플링 레이트와 다른 경우에만 리샘플링
            if sr != target_sr:
                print("Not resampled")
                #y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                # 덮어쓰기 저장
                #sf.write(file_path, y_resampled, target_sr)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path} - {e}")

# 사용 예시
resample_wav_folder_recursive("/data2/personal/sungjin/korean_dialects/modified")  # 폴더 경로 입력