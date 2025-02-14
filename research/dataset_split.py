import os
import random
import json

# wav 폴더 경로 설정
root_dir = '/data2/personal/sungjin/korean_standard/wav'

train_len = 0
valid_len = 0
test_len = 0

def add_train_len(path):
    global train_len
    with open(path) as file:
        data = json.load(file)
        length = data["기타정보"]["SpeechEnd"] - data["기타정보"]["SpeechStart"]
        train_len += length

def add_valid_len(path):
    global valid_len
    with open(path) as file:
        data = json.load(file)
        length = data["기타정보"]["SpeechEnd"] - data["기타정보"]["SpeechStart"]
        valid_len += length

def add_test_len(path):
    global test_len
    with open(path) as file:
        data = json.load(file)
        length = data["기타정보"]["SpeechEnd"] - data["기타정보"]["SpeechStart"]
        test_len += length

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

# 결과 파일 초기화
with open('trainset.txt', 'w') as train_file, \
     open('validset.txt', 'w') as valid_file, \
     open('testset.txt', 'w') as test_file:

    # wav 폴더 내 모든 하위 폴더 및 파일 순회
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.relpath(os.path.join(subdir, file), root_dir)
                rand_val = random.random()
                if rand_val < 0.28:
                    # 확률에 따라 파일 분배
                    rand_val2 = random.random()
                    if rand_val2 < 0.8:
                        train_file.write(relative_path + '\n')
                        add_train_len('/data2/personal/sungjin/korean_standard/label/' + relative_path.replace('.wav', '.json'))
                    elif rand_val2 < 0.9:
                        valid_file.write(relative_path + '\n')
                        add_valid_len('/data2/personal/sungjin/korean_standard/label/' + relative_path.replace('.wav', '.json'))
                    else:
                        test_file.write(relative_path + '\n')
                        add_test_len('/data2/personal/sungjin/korean_standard/label/' + relative_path.replace('.wav', '.json'))

print("train_len : " + str(seconds_to_hms(train_len)))
print("valid_len : " + str(seconds_to_hms(valid_len)))
print("test_len : " + str(seconds_to_hms(test_len)))