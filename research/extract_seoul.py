import zipfile
import json
import os
import glob
import random

# ZIP 파일 경로와 추출할 폴더 지정
directory = "/data2/personal/sungjin/korean_standard/TL1.zip"  # 압축 파일 경로
extract_path = "/data2/personal/sungjin/korean_standard/label"  # 압축 해제할 폴더
os.makedirs(extract_path, exist_ok=True)

# ZIP 파일 열기
with zipfile.ZipFile(directory, "r") as zip_ref:
    # ZIP 내 파일 목록 확인
    for file_name in zip_ref.namelist():
        if file_name.endswith(".json"):  # JSON 파일만 필터링
            with zip_ref.open(file_name) as file:
                try:
                    # JSON 파일 파싱
                    data = json.load(file)

                    # region 속성이 "Seoul"인지 확인
                    if data["화자정보"]["Region"] == "Seoul":
                        length = data["기타정보"]["SpeechEnd"] - data["기타정보"]["SpeechStart"]
                        if (length >= 1.00) and (length <= 6.00) and random.choice([True, False]):
                            zip_ref.extract(file_name, extract_path)

                except json.JSONDecodeError:
                    print(f"{file_name} JSON 파싱 오류 발생!")