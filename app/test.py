import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO

def plot_audio_with_speech_segments(audio_bytes: bytes, segments: list, target_sr: int = 16000):
    # Load the audio from bytes
    with BytesIO(audio_bytes) as audio_file:
        audio, sr = sf.read(audio_file)
    
    
    # Create time axis
    time = np.arange(0, len(audio)) / sr

    # Initialize plot
    plt.figure(figsize=(15, 5))
    plt.plot(time, audio, color='black', label='Audio Waveform')

    # Initialize variable to keep track of the current position
    current_pos = 0.0

    # Sort the segments to ensure they are in order
    segments = sorted(segments, key=lambda x: x[0])

    for segment in segments:
        start, end = segment
        # Plot Non-Speech segment before the current speech segment
        if current_pos < start:
            plt.axvspan(current_pos, start, color='red', alpha=0.5)
        # Plot Speech segment
        plt.axvspan(start, end, color='green', alpha=0.5)
        current_pos = end

    # Plot remaining Non-Speech segment after the last speech segment
    if current_pos < time[-1]:
        plt.axvspan(current_pos, time[-1], color='red', alpha=0.5)

    # Create custom legends
    import matplotlib.patches as mpatches
    speech_patch = mpatches.Patch(color='green', alpha=0.5, label='Speech Segment')
    nonspeech_patch = mpatches.Patch(color='red', alpha=0.5, label='Non-Speech Segment')
    plt.legend(handles=[speech_patch, nonspeech_patch])

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform with Speech and Non-Speech Segments')
    plt.show()

import requests

# FastAPI 서버 주소
API_URL = "http://localhost:8000/predict/"

# Audio file path
audio_file_path = "test_data/202407171343_test_4dca507c-aab2-4a55-ba9c-db27f0c65871_DESKTOP-RGVCDB1.wav"

# Load audio file
with open(audio_file_path, "rb") as f:
    audio_bytes = f.read()

params = {
    "window_length_in_sec": 0.025,
    "shift_length_in_sec": 0.01,
    "threshold": 0.3,
    "threshold_decay": 0.1,
    "speech_min_duration": 80,
    "noise_min_duration": 50,
    "speech_pad": 50,
    "target_sr": 16000
}

# 엔드포인트에 요청 보내기
response = requests.post(
    API_URL,
    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    data=params
)

# 응답 확인
if response.status_code == 200:
    result = response.json()
    speech_segments = result["speech_segments"]
    print("Detected Speech Segments:", speech_segments)
    
    # 시각화 함수 호출
    plot_audio_with_speech_segments(audio_bytes, speech_segments)
else:
    print("Error:", response.text)
