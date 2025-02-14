import json
import glob
import os

directory = "/data2/personal/sungjin/korean_dialects/modified/Jeolla/label"
wav_directory = "/data2/personal/sungjin/korean_dialects/modified/Jeolla/wav"
files = glob.glob(os.path.join(directory, "*.json"))

p_num = 0
nop_bad_len_num = 0
nop_good_len_num = 0
nop_good_len = 0

def contains_alphabet(s):
    return bool(set(s) & set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))

def contains_digit(s):
    return bool(set(s) & set("0123456789"))

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

punc = '"#$%&\'*+:;<=>@[\\]^_`{|}()/'
def contains_punctuation(s):
    return any(char in punc for char in s)

for file in files:
    with open(file, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
        id = data["id"]
        for utt in data["utterance"]:
            #if any(char in utt["dialect_form"] for char in ['/']):
            #    print("standard : " + utt["standard_form"])
            #    print("dialect : " + utt["dialect_form"])
            #    print(file)
            if any(char in utt["dialect_form"] for char in ['{', '&', '@', 'x', 'X']) or contains_punctuation(utt["dialect_form"]) or contains_digit(utt["dialect_form"]) or contains_alphabet(utt["dialect_form"]) or ('(())' in utt["dialect_form"]):
                p_num += 1
            else:
                length = utt["end"] - utt["start"]
                path = id + "/" + utt["id"].replace('.', '_') + ".wav"
                if (length >= 1) and (length <= 6) and os.path.exists(os.path.join(wav_directory, path)):
                    nop_good_len += length
                    nop_good_len_num += 1
                else:
                    nop_bad_len_num += 1


print("p_num : " + str(p_num)) # 아마 버려야할 데이터들
print("nop_bad_len_num : " + str(nop_bad_len_num))
print("nop_good_len_num : " + str(nop_good_len_num))
print("nop_good_len : " + str(seconds_to_hms(nop_good_len)))