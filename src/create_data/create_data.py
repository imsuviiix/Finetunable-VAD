import os
from omegaconf import OmegaConf
import subprocess

def main(type:str):
    tmp = 'src'
    data_folder = 'old_data'
    our_script = os.path.join(tmp, 'our_process_vad_data.py')
    google_data_root = os.path.join(data_folder, 'google_dataset_v2')
    our_speech_data_root = os.path.join(data_folder, 'speech_data')#for libri_Speech(.flac)
    our_background_data_root = os.path.join(data_folder, 'noise_data')
    if type == 'origin':
        out_dir = os.path.join(data_folder, 'our_manifest')
    elif type == 'sub':
        out_dir = os.path.join(data_folder, 'our_manifest','sub_manifest')

    command = [
    "python", our_script,  # 스크립트 이름을 포함합니다.
    "--out_dir", out_dir,
    "--speech_data_root", our_speech_data_root,
    "--background_data_root", our_background_data_root,
    "--log",
    "--rebalance_method", 'over',
    "--google_data_root", google_data_root#,
    # "--demo",
    # "--demo_rate", '0.05'
    ]

    subprocess.run(command)


if __name__ == '__main__':
    main('origin')


  