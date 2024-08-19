from argparse import ArgumentParser

def get_args(args=None):
    parser = ArgumentParser(description='VAD for .Wav file')
    parser.add_argument("--model_path", required=False, default="nemo_experiments/marblenet/2024-08-01_09-25-02/checkpoints/marblenet--val_loss=0.0095-epoch=13.ckpt", type=str)
    parser.add_argument("--threshold", required=False, default=0.50, type=float)
    parser.add_argument("--speech_min_duration", required=False, default=150, type=int, help='ms')
    parser.add_argument("--noise_min_duration", required=False, default=50, type=int, help='ms')
    parser.add_argument("--speech_pad", required=False, default=50, type=int, help='ms')
    parser.add_argument("--window_length_in_sec", required=False, default=0.025, type=float, help='ms')  # Shorter window length
    parser.add_argument("--shift_length_in_sec", required=False, default=0.01, type=float, help='ms')  # Shorter shift length
    parser.add_argument("--target_sr", required=False, default=16000, type=int, help='Hz')
    parser.add_argument("--audio_filepath", required=True, default=None, type=str, help='audio_path')

    if args is None:
        params = parser.parse_args()
    else:
        params = parser.parse_args(args)


    return params