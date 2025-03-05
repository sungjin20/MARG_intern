import os
from tqdm import tqdm
import numpy as np
import json
from TTS.tts.utils.text.cleaners import koreanCleaner
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text.characters import Graphemes
from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer

dialect_data_dir = '/data2/personal/sungjin/korean_dialects/modified'
standard_data_dir = '/data2/personal/sungjin/korean_standard'
KOREAN_PHONEMES = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ"
korean_punctuations = ".,!?~ "

gp = Graphemes(
    pad="_",
    eos="&",
    bos="*",
    blank=None,
    characters=KOREAN_PHONEMES,
    punctuations=korean_punctuations,
    is_unique=True,
    is_sorted=True
)

phonemizer = KO_KR_Phonemizer()
tokenizer = TTSTokenizer(characters=gp, phonemizer=phonemizer)

region = "Jeolla"
mode = "valid"

if region == 'standard':
    txt_file = os.path.join(standard_data_dir, f'standard_speech_{mode}set.txt')
    base_dir = os.path.join(standard_data_dir, 'label')
else:
    txt_file = os.path.join(dialect_data_dir, region, f"{region}_speech_{mode}set.txt")
    base_dir = os.path.join(dialect_data_dir, region, 'label')

with open(txt_file, 'r') as file:
    lines = file.readlines()

phoneme_folder = "/data2/personal/sungjin/korean_standard/phoneme"
os.makedirs(phoneme_folder, exist_ok=True)

for line in tqdm(lines, desc="Make phoneme from WAV files", unit="file"):
    line = line.strip()
    wav_path = os.path.join(base_dir, line)

    # 하위 폴더 구조를 유지하며 임베딩 파일 경로 설정
    relative_path = os.path.relpath(os.path.dirname(wav_path), base_dir)
    phoneme_subfolder = os.path.join(phoneme_folder, relative_path)
    phoneme_file_path = os.path.join(phoneme_subfolder, f"{os.path.splitext(os.path.basename(wav_path))[0]}.npy")

    # 이미 임베딩 파일이 존재하면 건너뛰기
    if os.path.exists(phoneme_file_path):
        continue

    if region == 'standard':
        text_path = wav_path.replace(".wav", ".json")
        with open(text_path, "r", encoding="utf-8") as f:
            data2 = json.load(f)
            text = data2["전사정보"]["TransLabelText"]
    else:
        text_path = os.path.join(base_dir, line.split('/')[0]) + ".json"
        with open(text_path, "r", encoding="utf-8-sig") as f:
            data2 = json.load(f)
            target_id = line.split('/')[1].split('.')[0].replace('_', '.')
            element = next((d for d in data2["utterance"] if d["id"] == target_id), None)
            text = element["dialect_form"]
            text = koreanCleaner(text)

    text = phonemizer.phonemize(text)
    ids = tokenizer.text_to_ids(text)

    # 임베딩 저장할 폴더 생성
    os.makedirs(phoneme_subfolder, exist_ok=True)

    # 임베딩 파일 저장
    np.save(phoneme_file_path, ids)