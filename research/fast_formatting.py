from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig
import json

train_json_path = "/data2/personal/sungjin/quick_load_samples/Jeolla_train_samples.json"
eval_json_path = "/data2/personal/sungjin/quick_load_samples/Jeolla_eval_samples.json"

data_config = BaseDatasetConfig(
    formatter="kdd",
    dataset_name="korean_dialects_dataset_Jeolla",
    meta_file_train="Jeolla_speech_trainset.txt",
    meta_file_val="Jeolla_speech_validset.txt",
    path='/data2/personal/sungjin/korean_dialects/modified/Jeolla',
    language="Jeolla",
)

train_samples, eval_samples = load_tts_samples(
    [data_config],
    eval_split=True,  # 학습 및 평가 데이터를 분리
)

with open(train_json_path, "w", encoding="utf-8") as train_file:
    json.dump(train_samples, train_file, ensure_ascii=False, indent=4)

with open(eval_json_path, "w", encoding="utf-8") as eval_file:
    json.dump(eval_samples, eval_file, ensure_ascii=False, indent=4)

print(f"✅ JSON 파일 저장 완료: \n  - {train_json_path}\n  - {eval_json_path}")