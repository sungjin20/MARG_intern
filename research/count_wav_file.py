import os
import random

def split_wav_files_train_valid(folder_path, top_n=30, train_file="train.txt", valid_file="valid.txt"):
    folder_wav_counts = {}

    # 최상위 폴더 탐색
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if os.path.isdir(subfolder_path):
            # 해당 폴더 내 .wav 파일 목록 가져오기
            wav_files = [file for file in os.listdir(subfolder_path) if file.lower().endswith('.wav')]
            folder_wav_counts[subfolder] = wav_files

    # 파일 개수가 적은 30개 폴더 선택
    sorted_folders = sorted(folder_wav_counts.items(), key=lambda x: len(x[1]))[:top_n]
    least_folders = set(folder for folder, _ in sorted_folders)  # 제외할 폴더 목록

    # train.txt, valid.txt 초기화
    with open(train_file, "w") as train_f, open(valid_file, "w") as valid_f:
        for folder, files in folder_wav_counts.items():
            if folder not in least_folders:  # 적은 30개 폴더 제외
                for file in files:
                    relative_path = os.path.join(folder, file)
                    
                    # 70% 확률로 train.txt, 30% 확률로 valid.txt
                    if random.random() < 0.7:
                        train_f.write(relative_path + "\n")
                    else:
                        valid_f.write(relative_path + "\n")

    print(f"파일이 랜덤하게 '{train_file}'(70%)과 '{valid_file}'(30%)로 저장되었습니다.")

# 사용 예시
split_wav_files_train_valid("/data2/personal/sungjin/korean_standard/wav", top_n=30)
