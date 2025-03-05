import os
import numpy as np
from tqdm import tqdm  # tqdm 임포트

# 루트 디렉토리 설정
root_dir = "/data2/personal/sungjin/korean_dialects/modified/Jeju/embedding"

# 디렉토리 개수 세기
total_dirs = sum([1 for _, dirs, _ in os.walk(root_dir) if dirs])

# 디렉토리 개수 출력
print(f"총 {total_dirs}개의 디렉토리가 있습니다.")

# 전체 하위 폴더와 파일을 순회하며 tqdm으로 진행 상태 표시
for subdir, _, files in tqdm(os.walk(root_dir), desc="Processing directories", total=total_dirs, unit="dir"):
    for file in files:
        if file.endswith(".npy"):  # .npy 파일만 선택
            file_path = os.path.join(subdir, file)
            try:
                data = np.load(file_path)  # .npy 파일 로드
                
                # shape가 (192,)가 아닌 경우만 출력
                if data.shape != (192,):
                    print(f"파일: {file_path}, shape: {data.shape}")
            except Exception as e:
                print(f"파일 {file_path} 로드 중 오류 발생: {e}")
