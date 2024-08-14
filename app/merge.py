import torch
import soundfile as sf
from torchaudio.transforms import Resample
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import onnxruntime as ort
from io import BytesIO


app = FastAPI()

# ONNX 모델 로드
def load_model(vad_model_path="../model/dcai_vad_v1.onnx"):
    try:
        main_session = ort.InferenceSession(vad_model_path)
        return main_session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX models: {str(e)}")

# Preprocessing audio
def preprocess_audio(audio_bytes, sample_rate=16000):
    try:
        with BytesIO(audio_bytes) as audio_file:
            audio, sr = sf.read(audio_file)
            print(f"Original audio length: {len(audio)} samples, Original sample rate: {sr}")

            if sr != sample_rate:
                resampler = Resample(sr, sample_rate)
                audio = resampler(torch.tensor(audio).float())
                print(f"Resampled audio length: {len(audio)} samples, Target sample rate: {sample_rate}")
            else:
                audio = torch.tensor(audio).float()

            return audio
    except Exception as e:
        raise ValueError(f"Failed to preprocess audio: {str(e)}")
    
def chunk_audio(audio_tensor, window_length_in_sec, shift_length_in_sec, sample_rate):
    window_length = int(window_length_in_sec * sample_rate)
    shift_length = int(shift_length_in_sec * sample_rate)
    audio_length = audio_tensor.shape[0]

    chunks = []
    for start in range(0, audio_length - window_length + 1, shift_length):
        end = start + window_length
        audio_chunk = audio_tensor[start:end].unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions
        chunks.append(audio_chunk.numpy())
    
    return chunks    


# Run main inference session
# Run inference on each chunk and detect speech segments
def run_inference_on_chunks(session, chunks, device, threshold, threshold_decay, 
                            speech_min_duration, noise_min_duration, 
                            speech_pad, window_length_in_sec, shift_length_in_sec, target_sr):
    frame_probs = []
    mfcc_frames = len(chunks)

    for frame_index, chunk in enumerate(chunks):
        inputs = {session.get_inputs()[0].name: chunk}

        with torch.no_grad():
            log_probs = session.run(None, inputs)[0]
            log_probs = torch.tensor(log_probs).to(device)  # Ensures log_probs is moved to the correct device
            probs = torch.softmax(log_probs, dim=-1)
            prob_speech = probs[:, 1].item()  # Get the probability of the speech class

            # Calculate the time for the current frame
            start_time = frame_index * shift_length_in_sec
            end_time = start_time + window_length_in_sec
            if end_time > mfcc_frames * shift_length_in_sec:
                end_time = (mfcc_frames - 1) * shift_length_in_sec
            frame_probs.append((start_time, end_time, prob_speech))
    
    
    # Post-process the frame probabilities to detect speech segments
    threshold_speech = threshold
    speech_segments = []
    is_speech = False
    segment_start = 0
    for start, end, prob in frame_probs:
        if prob >= threshold_speech:
            if not is_speech:
                is_speech = True
                segment_start = start
                threshold_speech = threshold_decay
        else:
            if is_speech:
                is_speech = False
                if end - segment_start >= speech_min_duration / 1000.0:
                    speech_segments.append((max(0, segment_start - speech_pad / 1000.0), end + speech_pad / 1000.0))
                    threshold_speech = threshold

    # Handle the case where the audio ends while in a speech segment
    if is_speech:
        if end - segment_start >= speech_min_duration / 1000.0:
            speech_segments.append((max(0, segment_start - speech_pad / 1000.0), end + speech_pad / 1000.0))

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

    return merged_segments  # Return the detected speech segments




# FastAPI app and endpoints
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...),
                        window_length_in_sec: float = Form(0.025),  
                        shift_length_in_sec: float = Form(0.01),    
                        threshold: float = Form(0.5),               
                        threshold_decay: float = Form(0.35),        
                        speech_min_duration: int = Form(150),       
                        noise_min_duration: int = Form(50),         
                        speech_pad: int = Form(50),                 
                        target_sr: int = Form(16000)                
):
    
    try:
        # 모델 로드
        model = '../model/dcai_vad_v1.onnx'
        main_session = load_model(model)
        logger.info("Models loaded successfully.")

        # 오디오 전처리
        audio_bytes = await file.read()
        audio_tensor = preprocess_audio(audio_bytes, sample_rate=target_sr)
        logger.info("Audio preprocessed successfully.")

        chunks = chunk_audio(audio_tensor, window_length_in_sec, shift_length_in_sec, target_sr)
        logger.info(f"Audio chunked into {len(chunks)} segments.")


        # speech 구간 검출
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        speech_segments = run_inference_on_chunks(main_session, chunks, device,
                                            window_length_in_sec=window_length_in_sec,
                                            shift_length_in_sec = shift_length_in_sec,
                                            threshold=threshold,
                                            threshold_decay=threshold_decay,
                                            speech_min_duration=speech_min_duration,
                                            noise_min_duration=noise_min_duration,
                                            speech_pad=speech_pad,
                                            target_sr=target_sr)
        
        logger.info("Inference completed successfully.")


        return {"speech_segments": speech_segments}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
