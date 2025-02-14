import os
import json

LABEL_DIR = "/data2/personal/sungjin/korean_standard/label"  # 루트 폴더 경로 설정

def contains_special_char(text):
    special_chars = set("@#$%^&*()-_=+[]{};:'\"\\|<>/`~") # use : '?!,.'
    return any(char in special_chars for char in text)

for root, _, files in os.walk(LABEL_DIR):
    for file in files:
        if file.endswith(".json"):  # JSON 파일만 선택
            json_path = os.path.join(root, file)

            # JSON 파일 열기
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data["전사정보"]["TransLabelText"]
                if contains_special_char(text): print(text)
