import zipfile
import os
import glob

# ZIP 파일 경로와 추출할 폴더 지정
directory = "/data2/personal/sungjin/korean_standard/temp/TS1.zip"  # 압축 파일 경로
extract_path = "/data2/personal/sungjin/korean_standard/wav"  # 압축 해제할 폴더
os.makedirs(extract_path, exist_ok=True)
directory2 = "/data2/personal/sungjin/korean_standard/label"  # 라벨 파일 경로
files = glob.glob(os.path.join(directory2, "**/*.json"))
json_filenames = [os.path.splitext(os.path.basename(file))[0] for file in files]

# ZIP 파일 열기
with zipfile.ZipFile(directory, "r") as zip_ref:
    # ZIP 내 파일 목록 확인
    for file_name in zip_ref.namelist():
        if file_name.endswith(".wav") and (os.path.splitext(os.path.basename(file_name))[0] in json_filenames):
            zip_ref.extract(file_name, extract_path)
            print("Extracted : " + file_name)

print("Finished!!!")