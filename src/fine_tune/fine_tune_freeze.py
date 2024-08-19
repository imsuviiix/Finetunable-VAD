import argparse
import os
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager
import nemo.collections.asr as nemo_asr

def get_args(args=None):
    parser = argparse.ArgumentParser(description='model training preprocess')
    parser.add_argument("--config_path", required=False, default='configs\marblenet_3x2x64.yaml', type=str)
    parser.add_argument("--manifest_path", required=False, default='old_data\our_manifest', type=str)
    parser.add_argument("--train_path", required=False, default='old_data\our_manifest\\fine_tune\\train_kaggle.json', type=str)
    parser.add_argument("--validation_path", required=False, default='old_data\our_manifest\\fine_tune\\val_kaggle.json', type=str)
    parser.add_argument("--test_path", required=False, default='old_data\our_manifest\\fine_tune\\test_kaggle.json', type=str)
    
    if args is None:
        config = parser.parse_args()
    else:
        config = parser.parse_args(args)

    return config

def set_dataset(train, val, test):
    train_dataset = train
    val_dataset = val
    test_dataset = test

    return train_dataset, val_dataset, test_dataset

def main():
    args = [
        "--train_path", 'old_data\\our_manifest\\fine_tune\\train_kaggle.json',
        "--validation_path", 'old_data\\our_manifest\\fine_tune\\val_kaggle.json',
        "--test_path", 'old_data\\our_manifest\\fine_tune\\test_kaggle.json'
    ]

    vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(os.path.join('../fine_tune', "fine_tuned_model.nemo"))

    arg = get_args(args)
    config_path = 'nemo_experiments\\marblenet\\2024-07-19_08-58-13\\hparams.yaml'
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    
    train_dataset, val_dataset, test_dataset = set_dataset(arg.train_path, arg.validation_path, arg.test_path)
    
    config.cfg.train_ds.manifest_filepath = train_dataset
    config.cfg.validation_ds.manifest_filepath = val_dataset
    config.cfg.test_ds.manifest_filepath = test_dataset

    accelerator = 'gpu'  # if torch.cuda.is_available() else 'cpu'
    config.cfg.trainer.devices = 1
    config.cfg.trainer.accelerator = accelerator

    config.cfg.trainer.max_epochs = 20
    config.cfg.optim.name = "adam"
    config.cfg.optim.lr = 0.001
    config.cfg.optim.sched.min_lr = 0.0001
    config.cfg.optim.momentum = 0.9  # Add this line
    config.cfg.train_ds.batch_size = 128
    config.cfg.validation_ds.batch_size = 128
    config.cfg.trainer.strategy = 'auto'

    print(OmegaConf.to_yaml(config))

    def freeze_layers(encoder, freeze_until_block_index):
    
        for i, block in enumerate(encoder.children()):
            if i < freeze_until_block_index:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

    # 예를 들어, 처음 3개의 JasperBlock 레이어만 동결하고 나머지 레이어는 학습하도록 설정
    freeze_layers(vad_model.encoder, freeze_until_block_index=4)
        # 디코더의 모든 파라미터 학습



    # Optimizer 설정
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, vad_model.parameters()), lr=config.cfg.optim.lr, momentum=config.cfg.optim.momentum, weight_decay=config.cfg.optim.weight_decay)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vad_model.parameters()), betas=(0.9,0.99), lr=config.cfg.optim.lr, weight_decay=config.cfg.optim.weight_decay)

    # Scheduler 설정
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    vad_model.setup_training_data(train_data_config=config.cfg.train_ds)
    vad_model.setup_validation_data(val_data_config=config.cfg.validation_ds)
    vad_model.setup_test_data(test_data_config=config.cfg.test_ds)

    # Trainer 설정
    trainer = pl.Trainer(
        max_epochs=config.cfg.trainer.max_epochs, 
        devices=config.cfg.trainer.devices, 
        accelerator=config.cfg.trainer.accelerator, 
        strategy=config.cfg.trainer.strategy,
        logger=False,  # Disable the default logger
        enable_checkpointing=False  # Disable default checkpointing
    )

    config.cfg.exp_manager.mlflow_logger_kwargs.tags = {"name": "test"}
    config.cfg.exp_manager.create_checkpoint_callback = True
    config.cfg.exp_manager.checkpoint_callback_params = {
        "monitor": "val_loss",
        "save_top_k": 5,
        "mode": "min",
        "save_last": True,
        "verbose": True,
    }
    exp_dir = exp_manager(trainer, config.cfg.get("exp_manager", None))
    exp_dir = str(exp_dir)

    # 모델 학습
    trainer.fit(vad_model)

if __name__ == '__main__':
    main()

