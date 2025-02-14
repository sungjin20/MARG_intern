import os
import json
import numpy as np
from tqdm import tqdm  # tqdm 추가
import re

# embedding 폴더 경로
embedding_dir = "/data2/personal/sungjin/korean_standard/embedding"

# 결과를 저장할 딕셔너리
result = {}

standard_trainset_file = "/data2/personal/sungjin/korean_standard/standard_speech_trainset.txt"
standard_validset_file = "/data2/personal/sungjin/korean_standard/standard_speech_validset.txt"

with open(standard_trainset_file, "r") as f:
    file_names = [line.strip().split('/')[1].replace('.wav', '.npy') for line in f.readlines()]
with open(standard_validset_file, "r") as f:
    file_names += [line.strip().split('/')[1].replace('.wav', '.npy') for line in f.readlines()]

# .npy 파일 경로 목록을 만들기
file_list = []
all_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(embedding_dir) for file in files]
with tqdm(total=len(all_files), desc="Scanning .npy files") as pbar:
    for file_path in all_files:
        if file_path.endswith(".npy") and os.path.basename(file_path) in file_names:  
            file_list.append(file_path)
        pbar.update(1)  # 진행 상태 업데이트

# tqdm 추가하여 진행 상태 표시
with tqdm(total=len(file_list), desc="Processing .npy files : standard") as pbar:
    for file_path in file_list:
        # .npy 파일 로드
        data = np.load(file_path).tolist()
        
        # 상대 경로를 이용한 키 생성
        relative_path = "korean_standard_dataset#wav/" + os.path.splitext(os.path.relpath(file_path, embedding_dir))[0]
        match = re.match(r"^(.*?_\w+)_\d+$", os.path.splitext(os.path.basename(file_path))[0]).group(1)
        
        # 결과 딕셔너리에 추가
        result[relative_path] = {
            "name": match,  # 파일명만 사용 (확장자 제외)
            "embedding": data
        }
        
        pbar.update(1)  # 진행 상태 업데이트

regions = ["Chungcheong", "Gangwon", "Gyeongsang", "Jeju", "Jeolla"]

for region in regions:
    embedding_dir = f"/data2/personal/sungjin/korean_dialects/modified/{region}/embedding"
    trainset_file = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_speech_trainset.txt"
    validset_file = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_speech_validset.txt"
    
    # 학습 및 검증 세트에서 사용할 파일 목록 읽기
    file_names = set()
    for file_path in [trainset_file, validset_file]:
        with open(file_path, "r") as f:
            file_names.update(line.strip().split('/')[1].replace('.wav', '.npy') for line in f.readlines())
    
    file_list = []
    all_files = [os.path.join(subdir, file) for subdir, _, files in os.walk(embedding_dir) for file in files]
    with tqdm(total=len(all_files), desc="Scanning .npy files") as pbar:
        for file_path in all_files:
            if file_path.endswith(".npy") and os.path.basename(file_path) in file_names:  
                file_list.append(file_path)
            pbar.update(1)  # 진행 상태 업데이트
    
    with tqdm(total=len(file_list), desc=f"Processing .npy files: {region}") as pbar:
        for file_path in file_list:
            try:
                emb = np.load(file_path).tolist()
                
                relative_path = f"korean_dialects_dataset_{region}#wav/" + os.path.splitext(os.path.relpath(file_path, embedding_dir))[0]
                json_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/label/" + os.path.splitext(os.path.relpath(file_path, embedding_dir))[0].split('/')[0] + ".json"
                
                with open(json_path, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                    target_id = os.path.splitext(os.path.relpath(file_path, embedding_dir))[0].split('/')[1].replace('_', '.')
                    element = next((d for d in data["utterance"] if d["id"] == target_id), None)
                    if element:
                        speaker_id = os.path.splitext(os.path.relpath(file_path, embedding_dir))[0].split('/')[0] + "_" + element["speaker_id"]
                        result[relative_path] = {"name": speaker_id, "embedding": emb}
                
                pbar.update(1)  # 진행 상태 업데이트

            except Exception as e:
                print(f"오류 발생: {e} - 파일: {file_path}")
                continue  # 오류가 발생해도 다음 파일로 넘어가도록 처리

# JSON 파일로 저장
json_path = "speakers_mixed.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"JSON 파일 저장 완료: {json_path}")
