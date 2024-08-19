import base64
import numpy as np
import io
from pydub import AudioSegment
import wave
SAMPLE_RATE = 16000
def save_bytearray_to_wav(byte_array, filename,
                          channels=1, sample_width=2, sample_rate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(byte_array)
def wav_to_numpy(wav_file):
    # Load the WAV file using pydub
    try:
        audio = AudioSegment.from_wav(wav_file)
        print("wav information")
        print(f'audio sample_rate: {audio.frame_rate}')
        print(f'audio channel: {audio.channels}')
        print(f'audio sample_width: {audio.sample_width}')
        # Convert the audio to numpy array
        if audio.frame_rate != 16000:
            raise ValueError(f"{audio.frame_rate} is Unsupported sampling rate. Only 16000 Hz is supported")
        if audio.channels != 1:
            raise ValueError(f"{audio.channels} is Unsupported channels. Only mono channel is supported")
        if audio.sample_width != 2:
            raise ValueError(f"{audio.sample_width} byte is Unsupported bit sample. Only 2byte is supported")
        samples = np.array(
            audio.get_array_of_samples()).astype(np.float32) / 32768.
        return samples
    except ValueError as ve:
        print("Not suportted Wav Format: ", ve)
def load_audio_from_file(file: str, sr: int = SAMPLE_RATE):
        """
        Open an audio file and read as mono waveform, resampling as necessary
        Parameters
        ----------
        file: str
            The audio file to open
        sr: int
            The sample rate to resample the audio if necessary
        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
def read_audio(audio, type: str):
    if type == 'pcm':
        decode_string = base64.b64decode(audio)
        direct_audio = (
                np.frombuffer(decode_string, np.int16).
                flatten().astype(np.float32)
                / 32768.0
            )
        audio_time = len(direct_audio)
    elif type == "format":
        decode_string = base64.b64decode(audio)
        direct_audio = wav_to_numpy(io.BytesIO(decode_string))
        audio_time = len(direct_audio)
    elif type == "file":
        if isinstance(audio, bytes):
            audio = base64.b64decode(audio)
            audio = audio.decode("utf-8")
        direct_audio = audio
        audio_time = len(load_audio_from_file(audio))
    return direct_audio, audio_time
def float32_to_pcm16(float_array):
    # Ensure the input is a NumPy array of float32
    float_array = np.asarray(float_array, dtype=np.float32)
    # Scale the float array to the range of int16
    int16_array = np.int16(float_array * 32767)
    # Convert the int16 array to bytes
    pcm_bytes = int16_array.tobytes()
    return pcm_bytes