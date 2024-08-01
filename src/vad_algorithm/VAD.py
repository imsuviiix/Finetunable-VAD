import torch
from tqdm import tqdm
import os
import nemo.collections.asr as nemo_asr
from torchaudio.transforms import Resample
import soundfile as sf
from args import get_args
import matplotlib.pyplot as plt
import numpy as np

def load_wav(filepath: str, target_sr: int = 16000) -> torch.Tensor:
    audio, sr = sf.read(filepath)
    if sr != target_sr:
        resampler = Resample(sr, target_sr)
        audio = resampler(torch.tensor(audio).float())
    return torch.tensor(audio).float(), sr

def init_model(model_path:str):
    if model_path.endswith('.nemo'):
        vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(restore_path=model_path)
    elif model_path.endswith('.ckpt'):
        vad_model = nemo_asr.models.EncDecClassificationModel.load_from_checkpoint(checkpoint_path=model_path)
    else:
        vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name=model_path)
    return vad_model

def plot_audio_with_speech_segments(audio_filepath: str, segments: list, target_sr: int = 16000):
    # Load the audio file
    audio, sr = sf.read(audio_filepath)
    if sr != target_sr:
        raise ValueError("Sample rate of the audio file does not match the target sample rate.")
    
    # Create time axis
    time = np.arange(0, len(audio)) / sr

    # Initialize plot
    plt.figure(figsize=(15, 10))
    plt.plot(time, audio, color='black', label='Audio Waveform')

    # Plot non-speech segments
    prev_end = 0
    for start, end in segments:
        if start > prev_end:
            plt.axvspan(prev_end, start, color='red', alpha=0.5, label='Non-Speech Segment')
        plt.axvspan(start, end, color='green', alpha=0.5, label='Speech Segment')
        prev_end = end

    # Plot any remaining non-speech segment at the end of the audio
    if prev_end < time[-1]:
        plt.axvspan(prev_end, time[-1], color='red', alpha=0.5, label='Non-Speech Segment')

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform with Speech and Non-Speech Segments')
    plt.xticks(size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def main():

    args = ['--audio_filepath', 'SDH_DATA/검토필요_wav/전문 용어/202407161039_test_35163392-cb87-4d40-bb07-20b6e0ffca8f_DESKTOP-RGVCDB1.wav']

    arg = get_args(args)
    model_path = arg.model_path
    window_length_in_sec = arg.window_length_in_sec
    shift_length_in_sec = arg.shift_length_in_sec
    threshold = arg.threshold
    speech_min_duration = arg.speech_min_duration
    noise_min_duration = arg.noise_min_duration
    speech_pad = arg.speech_pad
    target_sr = arg.target_sr
    audio_filepath = arg.audio_filepath

    vad_model = init_model(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the VAD model
    vad_model.to(device)
    vad_model.eval()

    # Load the audio file
    audio, sr = load_wav(audio_filepath, target_sr=target_sr)
    audio_length = audio.shape[0]
    window_length = int(window_length_in_sec * target_sr)
    shift_length = int(shift_length_in_sec * target_sr)

    # Initialize lists to store start and end times with probabilities
    frame_probs = []

    # Chunk the audio and create batches
    chunks = []
    for start in range(0, audio_length - window_length + 1, shift_length):
        end = start + window_length
        audio_chunk = audio[start:end].unsqueeze(0)  # Add batch dimension
        chunks.append(audio_chunk)


    batch_size = 512  # Define your batch size
    num_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size != 0 else 0)

    # Perform inference on each batch
    for i in range(num_batches):
        batch = chunks[i * batch_size:(i + 1) * batch_size]
        batch = torch.cat(batch, dim=0).to(device)
        input_signal_length = torch.tensor([window_length] * batch.size(0)).long().to(device)

        with torch.no_grad():
            log_probs = vad_model(input_signal=batch, input_signal_length=input_signal_length)
            probs = torch.softmax(log_probs, dim=-1)
            pred = probs[:, 1]  # Get the probability of the speech class

            for j in range(pred.size(0)):
                start_time = (i * batch_size + j) * shift_length_in_sec
                end_time = start_time + window_length_in_sec
                frame_prob = pred[j].mean().item()  # Get the mean probability for the chunk
                frame_probs.append((start_time, end_time, frame_prob))
    # print(f"it is a probs: {frame_probs}")

    # Post-process the frame probabilities to detect speech segments
    speech_segments = []
    is_speech = False
    segment_start = 0
    for start, end, prob in frame_probs:
        if prob >= threshold:
            if not is_speech:
                is_speech = True
                segment_start = start
                threshold = 0.35
        else:
            if is_speech:
                is_speech = False
                if end - segment_start >= speech_min_duration / 1000.0:
                    speech_segments.append((max(0, segment_start - speech_pad / 1000.0), end + speech_pad / 1000.0))

        # Handle the case where the audio ends while in a speech segment
        if is_speech:
            if end - segment_start >= speech_min_duration / 1000.0:
                speech_segments.append((max(0, segment_start - speech_pad / 1000.0), end + speech_pad / 1000.0))
                threshold = arg.threshold

    # Merge speech segments with short noise in between
    merged_segments = []
    if speech_segments:
        current_start, current_end = speech_segments[0]
        for next_start, next_end in speech_segments[1:]:
            if next_start - current_end < noise_min_duration / 1000.0:
                current_end = next_end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_segments.append((current_start, current_end))

    # Print the detected speech segments
    for start, end in merged_segments:
        print(f"Speech segment: Start: {start:.4f}, End: {end:.4f}")

    plot_audio_with_speech_segments(audio_filepath, merged_segments, target_sr=target_sr)


if __name__ == "__main__":
    main()