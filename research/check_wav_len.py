import zipfile
import json
import os
import glob

# ZIP 파일 경로와 추출할 폴더 지정
directory = "/data2/personal/sungjin/korean_standard/label"  # 압축 파일 경로
files = glob.glob(os.path.join(directory, "**/*.json"))

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

wav_len = 0
sen_cnt = 0
spk_arr = []
man_num = 0
# ZIP 파일 열기
for file_name in files:
    with open(file_name) as file:
        try:
            # JSON 파일 파싱
            data = json.load(file)

            length = data["기타정보"]["SpeechEnd"] - data["기타정보"]["SpeechStart"]
            wav_len += length
            sen_cnt += 1
            if data["기본정보"]["NumberOfSpeaker"] not in spk_arr:
                spk_arr.append(data["기본정보"]["NumberOfSpeaker"])
                if data["화자정보"]["Gender"] == "Male": man_num += 1

        except json.JSONDecodeError:
            print(f"{file_name} JSON 파싱 오류 발생!")

print("==========================")
print("Tot_time = " + seconds_to_hms(int(wav_len)))
print("Data_cnt = " + str(sen_cnt))
print("Spk_num = " + str(len(spk_arr)))
print("man_woman_rate = " + str(man_num/len(spk_arr)))
print("==========================")