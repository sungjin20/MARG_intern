import json
import glob
import os
import random

region = "Chungcheong"
directory = "/data2/personal/sungjin/korean_dialects/modified/" + region + "/label"
wav_directory = "/data2/personal/sungjin/korean_dialects/modified/" + region + "/wav"
files = glob.glob(os.path.join(directory, "*.json"))

def contains_alphabet(s):
    return bool(set(s) & set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))

def contains_digit(s):
    return bool(set(s) & set("0123456789"))

def seconds_to_hms(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

punc = '"#$%&\'*+:;<=>@[\\]^_`{|}()/-~‘’😍ㄴ'
def contains_punctuation(s):
    return any(char in punc for char in s)

speech_train_len = 0
speech_valid_len = 0
speech_test_len = 0
classification_train_len = 0
classification_valid_len = 0
classification_test_len = 0

with open(region + '_speech_trainset.txt', 'w') as speech_train_file, \
     open(region + '_speech_validset.txt', 'w') as speech_valid_file, \
     open(region + '_speech_testset.txt', 'w') as speech_test_file, \
     open(region + '_classification_trainset.txt', 'w') as classification_train_file, \
     open(region + '_classification_validset.txt', 'w') as classification_valid_file, \
     open(region + '_classification_testset.txt', 'w') as classification_test_file:
    for file in files:
        with open(file, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            id = data["id"]
            for utt in data["utterance"]:
                if (utt["speaker_id"] is None) or contains_punctuation(utt["dialect_form"]) or contains_digit(utt["dialect_form"]) or contains_alphabet(utt["dialect_form"]) or ('(())' in utt["dialect_form"]):
                    pass
                else:
                    length = utt["end"] - utt["start"]
                    path = id + "/" + utt["id"].replace('.', '_') + ".wav"
                    if (length >= 1) and (length <= 6) and os.path.exists(os.path.join(wav_directory, path)):
                        rand_val = random.random()
                        if rand_val < 0.28436:
                            rand_val3 = random.random()
                            if rand_val3 < 0.5:
                                rand_val2 = random.random()
                                if rand_val2 < 0.8:
                                    speech_train_file.write(path + '\n')
                                    speech_train_len += length
                                elif rand_val2 < 0.9:
                                    speech_valid_file.write(path + '\n')
                                    speech_valid_len += length
                                else:
                                    speech_test_file.write(path + '\n')
                                    speech_test_len += length
                            else:
                                rand_val2 = random.random()
                                if rand_val2 < 0.66666:
                                    classification_train_file.write(path + '\n')
                                    classification_train_len += length
                                elif rand_val2 < 0.833333:
                                    classification_valid_file.write(path + '\n')
                                    classification_valid_len += length
                                else:
                                    classification_test_file.write(path + '\n')
                                    classification_test_len += length
                            
print("speech_train_len : " + str(seconds_to_hms(speech_train_len)))
print("speech_valid_len : " + str(seconds_to_hms(speech_valid_len)))
print("speech_test_len : " + str(seconds_to_hms(speech_test_len)))
print("classification_train_len : " + str(seconds_to_hms(classification_train_len)))
print("classification_valid_len : " + str(seconds_to_hms(classification_valid_len)))
print("classification_test_len : " + str(seconds_to_hms(classification_test_len)))