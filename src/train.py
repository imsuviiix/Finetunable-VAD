import argparse
from argparse import ArgumentParser
import os
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
import nemo
import nemo.collections.asr as nemo_asr
from pytorch_lightning.callbacks import ModelCheckpoint



def get_args(args=None):
    """
    config_path
    manifest_path
    ...

    """
    parser = argparse.ArgumentParser(description='model training preprocess')
    parser.add_argument("--config_path", required=False, default='configs\marblenet_3x2x64.yaml', type=str)
    parser.add_argument("--manifest_path", required=False, default='old_data\our_manifest', type=str)
    parser.add_argument("--train_path", required=False, default='old_data/our_manifest/sub_manifest/balanced_noise_training_manifest.json,old_data/our_manifest/sub_manifest/balanced_speech_training_manifest.json' , type=str)
    parser.add_argument("--validation_path", required=False, default='old_data/our_manifest/sub_manifest/balanced_noise_validation_manifest.json,old_data/our_manifest/sub_manifest/balanced_speech_validation_manifest.json' , type=str)
    parser.add_argument("--test_path", required=False, default='old_data/our_manifest/sub_manifest/balanced_noise_testing_manifest.json,old_data/our_manifest/sub_manifest/balanced_speech_testing_manifest.json' , type=str)
    
    
    if args is None:
        config = parser.parse_args()
    else:
        config = parser.parse_args(args)


    return config

def set_dataset(train,val,test):
    train_dataset = train
    val_dataset = val
    test_dataset = test

    return train, val, test


def main():
    args = [
     "--train_path", 'old_data/our_manifest/balanced_noise_training_manifest.json,old_data/our_manifest/balanced_speech_training_manifest.json',
     "--validation_path", 'old_data/our_manifest/balanced_noise_validation_manifest.json,old_data/our_manifest/balanced_speech_validation_manifest.json',
     "--test_path", 'old_data/our_manifest/balanced_noise_testing_manifest.json,old_data/our_manifest/balanced_speech_testing_manifest.json'
    ]

    arg = get_args(args)
    config_path = arg.config_path
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    labels = config.model.labels
    sample_rate = config.model.sample_rate

    train_dataset, val_dataset, test_dataset = set_dataset(arg.train_path,
                                                           arg.validation_path,
                                                           arg.test_path)
    
    config.model.train_ds.manifest_filepath = train_dataset
    config.model.validation_ds.manifest_filepath = val_dataset
    config.model.test_ds.manifest_filepath = test_dataset


    accelerator = 'gpu' #if torch.cuda.is_available() else 'cpu'
    config.trainer.devices = 1
    config.trainer.accelerator = accelerator

# Reduces maximum number of epochs to 5 for quick demonstration
    config.trainer.max_epochs = 1

# Remove distributed training flags
    config.trainer.strategy = 'auto'
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(**config.trainer)

    config.exp_manager.mlflow_logger_kwargs.tags = {"name":"test"}
    # config.exp_manager.mlflow_logger_kwargs.run_id = "Test"
    exp_dir = exp_manager(trainer, config.get("exp_manager", None))
    exp_dir = str(exp_dir)


    vad_model = nemo_asr.models.EncDecClassificationModel(cfg=config.model, trainer=trainer)

    trainer.fit(vad_model)

    return config, vad_model, trainer


# if __name__ == '__main__':

config, vad_model, trainer = main()
    
    