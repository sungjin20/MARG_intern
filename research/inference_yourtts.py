from TTS.utils.synthesizer import Synthesizer
import os

device = "cuda:11"
out_path = "/data2/personal/sungjin/demo"

tts_path = "/data2/personal/sungjin/korean_standard/YourTTS-ko-ft-softlabel-ckp28-February-24-2025_07+58AM-165e9d5a/checkpoint_440000.pth"
tts_config_path = "/data2/personal/sungjin/korean_standard/YourTTS-ko-ft-softlabel-ckp28-February-24-2025_07+58AM-165e9d5a/config.json"
speakers_file_path = None
language_ids_file_path = None
vocoder_path = None
vocoder_config_path = None
encoder_path = None
encoder_config_path = None
vc_path = None
vc_config_path = None
model_dir = None
voice_dir = None

text = "언제가 놀러가기가 가장 좋은 계절이라고 생각해?"
speaker_idx = "DKSR20003009_2"
language_idx_list = ["standard", "Chungcheong", "Gyeongsang", "Gangwon", "Jeju", "Jeolla"]
speaker_wav = None
reference_wav = None
capacitron_style_wav = None
capacitron_style_text = None
reference_speaker_idx = None

synthesizer = Synthesizer(
    tts_path,
    tts_config_path,
    speakers_file_path,
    language_ids_file_path,
    vocoder_path,
    vocoder_config_path,
    encoder_path,
    encoder_config_path,
    vc_path,
    vc_config_path,
    model_dir,
    voice_dir,
).to(device)

for i in range(20):
    wav = synthesizer.tts(
        text,
        speaker_name=speaker_idx,
        language_name="Gyeongsang",
        speaker_wav=speaker_wav,
        reference_wav=reference_wav,
        style_wav=capacitron_style_wav,
        style_text=capacitron_style_text,
        reference_speaker_name=reference_speaker_idx,
    )
    folder_path = out_path + "/" + text
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    out_file_path = folder_path + "/demo_Gyeongsang_" + str(i) + ".wav"
    print(" > Saving output to {}".format(out_file_path))
    synthesizer.save_wav(wav, out_file_path, pipe_out=False)