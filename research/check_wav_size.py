import os

# 파일 경로
region = "Jeolla"
wav_dir_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/wav"
label_dir_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/label"
trainset_path = f"/data2/personal/sungjin/korean_dialects/modified/{region}/{region}_speech_validset.txt"

# 총 파일 크기 저장 변수 (바이트 단위)
total_size = 0

# 파일 경로 읽기
with open(trainset_path, "r") as f:
    for line in f:
        wav_path = line.strip()  # 개행 문자 제거
        #if os.path.exists(os.path.join(label_dir_path, wav_path.replace(".wav", ".json"))):  # 파일 존재 여부 확인
        #    total_size += os.path.getsize(os.path.join(label_dir_path, wav_path.replace(".wav", ".json")))  # 파일 크기 추가
        if os.path.exists(os.path.join(wav_dir_path, wav_path)):  # 파일 존재 여부 확인
            total_size += os.path.getsize(os.path.join(wav_dir_path, wav_path))  # 파일 크기 추가

# 바이트 -> GB 변환 (1GB = 1024^3 바이트)
total_size_gb = total_size / (1024 ** 3)

# 결과 출력
print(f"총 음성 데이터 용량: {total_size_gb:.2f} GB")
