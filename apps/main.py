import torch
import soundfile as sf
from torchaudio.transforms import Resample
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import onnxruntime as ort
from io import BytesIO


app = FastAPI()

# ONNX 모델 로드
def load_model(preprocessor_model_path="../model/dcai_preprocessor.onnx", vad_model_path="../model/dcai_vad.onnx"):
    try:
        preprocessing_session = ort.InferenceSession(preprocessor_model_path)
        main_session = ort.InferenceSession(vad_model_path)
        return preprocessing_session, main_session
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

# Run preprocessing session
def run_preprocessing(preprocessing_session, audio_tensor):
    try:
        print(f"Preprocessed audio input length: {audio_tensor.shape[-1]} samples")
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        input_data = audio_tensor.numpy()
        inputs = {preprocessing_session.get_inputs()[0].name: input_data}
        
        # Log expected input shape
        expected_input_shape = preprocessing_session.get_inputs()[0].shape
        print(f"Model expected input shape: {expected_input_shape}")
        print(f"Provided input shape: {input_data.shape}")
        
        preprocessed_data = preprocessing_session.run(None, inputs)
        preprocessed_tensor = torch.tensor(preprocessed_data[0])
        
        print(f"Preprocessed data length: {preprocessed_tensor.shape[-1]} samples")
        print(f"Preprocessed data shape: {preprocessed_tensor.shape}")
        return preprocessed_tensor
    except Exception as e:
        raise RuntimeError(f"Failed during preprocessing: {str(e)}")

# Run main inference session
def run_main_inference(main_session, preprocessed_data_tensor, threshold: float = 0.5, threshold_decay: float = 0.35,
                    speech_min_duration: int = 150, noise_min_duration: int = 50,
                    speech_pad: int = 50, window_length_in_sec: float = 0.025, shift_length_in_sec: float = 0.01, target_sr: int = 16000):
    try: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # preprocessed_data_tensor는 이미 청킹된 MFCC 프레임들을 포함
        mfcc_frames = preprocessed_data_tensor.shape[-1]
        print(f"MFCC frame length: {mfcc_frames} frames")

        # Initialize lists to store start and end times with probabilities
        frame_probs = []

        # Perform inference on each frame (each time step in MFCC data)
        for frame_index in range(mfcc_frames):
            # Select the MFCC frame at the current time step

            # Check if end_idx exceeds the length of the audio data

            mfcc_frame = preprocessed_data_tensor[:, :, frame_index].unsqueeze(-1).to(device)
            # print(f"Processing MFCC frame {frame_index}, size: {mfcc_frame.shape}")

            with torch.no_grad():
                inputs = {main_session.get_inputs()[0].name: mfcc_frame.cpu().numpy()}
                log_probs = main_session.run(None, inputs)[0]
                log_probs = torch.tensor(log_probs).to(device)
                probs = torch.softmax(log_probs, dim=-1)
                prob_speech = probs[:, 1].item()  # Get the probability of the speech class

                # Calculate the time for the current frame
                start_time = frame_index *shift_length_in_sec
                end_time = start_time + window_length_in_sec
                if end_time > preprocessed_data_tensor.shape[2]*shift_length_in_sec:
                    end_time = (preprocessed_data_tensor.shape[2]-1)*shift_length_in_sec
                convert =target_sr*shift_length_in_sec
                frame_probs.append((start_time, end_time, prob_speech))
        print(frame_probs)  

        threshold_speech = threshold
        # Post-process the frame probabilities to detect speech segments
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

    except Exception as e:
        raise RuntimeError(f"Failed during main inference: {str(e)}")



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
        preprocessing_session, main_session = load_model()
        logger.info("Models loaded successfully.")

        # 오디오 전처리
        audio_bytes = await file.read()
        audio_tensor = preprocess_audio(audio_bytes, sample_rate=target_sr)
        logger.info("Audio preprocessed successfully.")

        # 전처리된 데이터를 사용하여 infer
        preprocessed_data_tensor = run_preprocessing(preprocessing_session, audio_tensor)
        logger.info("Preprocessing completed successfully.")

        # speech 구간 검출
        speech_segments = run_main_inference(main_session, preprocessed_data_tensor,
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
