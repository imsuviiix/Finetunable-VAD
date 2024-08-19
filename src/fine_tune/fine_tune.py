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
import librosa
import json
import IPython.display as ipd



def get_args(args=None):
    """
    config_path
    manifest_path
    ...

    """
    parser = argparse.ArgumentParser(description='model training preprocess')
    parser.add_argument("--config_path", required=False, default='configs\marblenet_3x2x64.yaml', type=str)
    parser.add_argument("--manifest_path", required=False, default='old_data\our_manifest', type=str)
    parser.add_argument("--train_path", required=False, default='old_data\our_manifest\\fine_tune\\train_kaggle.json' , type=str)
    parser.add_argument("--validation_path", required=False, default='old_data\our_manifest\\fine_tune\\val_kaggle.json' , type=str)
    parser.add_argument("--test_path", required=False, default='old_data\our_manifest\\fine_tune\\test_kaggle.json' , type=str)
    
    
    if args is None:
        config = parser.parse_args()
    else:
        config = parser.parse_args(args)


    return config

def set_dataset(train,val,test):
    train_dataset = train
    val_dataset = val
    test_dataset = test

    return train_dataset, val_dataset, test_dataset


def main():
    args = [
     "--train_path", 'old_data\our_manifest\\fine_tune\\train_kaggle.json',
     "--validation_path", 'old_data\our_manifest\\fine_tune\\val_kaggle.json',
     "--test_path", 'old_data\our_manifest\\fine_tune\\test_kaggle.json'
    ]

    vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(os.path.join('../fine_tune', "fine_tuned_model.nemo"))

    # checkpoint_path = 'nemo_experiments\marblenet\\2024-07-19_08-58-13\checkpoints\marblenet--val_loss=0.0021-epoch=17.ckpt'
    # checkpoint = torch.load(checkpoint_path)
    # vad_model.load_state_dict(checkpoint['state_dict'], strict=False)

    arg = get_args(args)
    config_path = 'nemo_experiments\marblenet\\2024-07-19_08-58-13\hparams.yaml'
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    labels = config.cfg.labels
    sample_rate = config.cfg.sample_rate
    
    train_dataset, val_dataset, test_dataset = set_dataset(arg.train_path,
                                                           arg.validation_path,
                                                           arg.test_path)
    
    config.cfg.train_ds.manifest_filepath = train_dataset
    config.cfg.validation_ds.manifest_filepath = val_dataset
    config.cfg.test_ds.manifest_filepath = test_dataset


    accelerator = 'gpu' #if torch.cuda.is_available() else 'cpu'
    config.cfg.trainer.devices = 1
    config.cfg.trainer.accelerator = accelerator

# Reduces maximum number of epochs to 5 for quick demonstration
    config.cfg.trainer.max_epochs = 5
    config.cfg.optim.lr = 0.0001
    config.cfg.optim.sched.min_lr = 0.00001
    config.cfg.train_ds.batch_size = 64
    # config.model.train_ds.num_worker = 1
    config.cfg.validation_ds.batch_size = 64
    # config.model.validation_ds.num_worker = 1
# Remove distributed training flags
    config.cfg.trainer.strategy = 'auto'
    
    print(OmegaConf.to_yaml(config))

    trainer = pl.Trainer(**config.cfg.trainer)

    config.cfg.exp_manager.mlflow_logger_kwargs.tags = {"name":"test"}
    # config.exp_manager.mlflow_logger_kwargs.run_id = "Test"
    exp_dir = exp_manager(trainer, config.cfg.get("exp_manager", None))
    exp_dir = str(exp_dir)


    vad_model = nemo_asr.models.EncDecClassificationModel(cfg=config.cfg, trainer=trainer)

    trainer.fit(vad_model)



if __name__ == '__main__':

    main()
    
    