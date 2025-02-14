import librosa
import soundfile as sf
from pydub import AudioSegment

def convert_and_resample(input_file, output_file, target_sr=16000):
    # Step 1: M4A → WAV 변환 (output_file에 바로 저장)
    audio = AudioSegment.from_file(input_file, format="m4a")
    audio.export(output_file, format="wav")

    # Step 2: 16kHz로 리샘플링
    y, sr = librosa.load(output_file, sr=None)  # 원본 샘플링 레이트 로드
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Step 3: output_file을 덮어쓰기
    sf.write(output_file, y_resampled, target_sr)
    print(f"변환 및 리샘플링 완료: {output_file}")

# 예제 사용
convert_and_resample("test.m4a", "test.wav")
