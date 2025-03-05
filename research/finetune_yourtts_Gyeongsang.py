import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
import json

# 실행을 위한 이름
RUN_NAME = "YourTTS-ko-ft-Gyeongsang"

# 모델 출력(구성, 체크포인트, 텐서보드 로그)을 저장할 경로
OUT_PATH = '/data2/personal/sungjin/korean_standard'  # "/raid/coqui/Checkpoints/original-YourTTS/"

RESTORE_PATH = '/data2/personal/sungjin/korean_standard/checkpoint_280000.pth'

# 학습 및 평가에 사용할 배치 크기를 여기에서 설정합니다.
BATCH_SIZE = 28

SAMPLE_RATE = 16000

# 학습에 사용되는 최대 오디오 길이(초 단위). 이보다 큰 오디오는 무시됩니다.
MAX_AUDIO_LEN_IN_SECONDS = 6

# 구성 초기화
Gyeongsang_data_config = BaseDatasetConfig(
    formatter="kdd",
    dataset_name="korean_dialects_dataset_Gyeongsang",
    meta_file_train="Gyeongsang_speech_trainset.txt",
    meta_file_val="Gyeongsang_speech_validset.txt",
    path='/data2/personal/sungjin/korean_dialects/modified/Gyeongsang',
    language="Gyeongsang",
)

DATASETS_CONFIG_LIST = [Gyeongsang_data_config]

D_VECTOR_FILES = []  # 학습 중에 사용할 스피커 임베딩/d-벡터 목록

D_VECTOR_FILES.append("/data2/personal/sungjin/korean_dialects/speakers_mixed.json")

# 학습에 사용되는 오디오 구성
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0,
    mel_fmax=8000,
    num_mels=80,
)

# YourTTS 모델에 필요한 매개변수를 설정하여 VITSArgs 초기화
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=192,
    num_layers_text_encoder=10,
    hidden_channels=196,
    num_layers_flow=4,
    resblock_type_decoder="1",
    # 다국어 학습을 활성화하는 데 유용한 매개변수
    use_language_embedding=True,
    embedded_language_dim=64,
    num_languages=6
)

phoneme_cache_folder_path = '/home/research/phoneme_cache_ft_Gyeongsang'

# 일반 학습 구성. 여기에서 배치 크기 및 기타 유용한 매개변수를 변경할 수 있음
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    print_step=25,
    plot_step=100,
    log_model_step=1000,
    save_step=20000,
    save_n_checkpoints=10,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=True,
    phonemizer="ko_kr_phonemizer",
    compute_input_seq_cache=True,
    add_blank=False,
    characters=CharactersConfig(
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ",
        punctuations=".,!?~ ",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=phoneme_cache_folder_path,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    test_sentences=[
        [
            "제 나이가 올해로 스물네살입니다.",
            "DKSR20000230_1",
            None,
            "Gyeongsang",
        ],
    ],
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
)

data_dir = "/data2/personal/sungjin/quick_load_samples"
regions = ["Gyeongsang"]
train_samples = []
eval_samples = []
for region in regions:
    train_path = os.path.join(data_dir, f"{region}_train_samples.json")
    eval_path = os.path.join(data_dir, f"{region}_eval_samples.json")
    with open(train_path, "r", encoding="utf-8") as train_file:
        train_samples += json.load(train_file)
    with open(eval_path, "r", encoding="utf-8") as eval_file:
        eval_samples += json.load(eval_file)

# 모델 초기화
model = Vits.init_from_config(config)

# 학습기 초기화 및 🚀 시작
trainer = Trainer(
    #TrainerArgs(continue_path='/data2/personal/sungjin/korean_standard/YourTTS-ko-standard-February-06-2025_04+11PM-0000000', gpu=2), # gpu번호 설정
    TrainerArgs(restore_path=RESTORE_PATH, gpu=5), # gpu번호 설정
    config,  # 모델 구성
    output_path=OUT_PATH,  # 출력 경로
    model=model,  # 모델 객체
    train_samples=train_samples,  # 학습 샘플
    eval_samples=eval_samples,  # 평가 샘플
)
trainer.fit()  # 학습 시작