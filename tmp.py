# python tests/trainer.py

import torch
from opt import Opts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from src.models import MODEL_REGISTRY
from ds import AIC22TextJsonDataset, AIC22TrackJsonWithMotionDataset
from pathlib import Path
from typing import List 


def train(model_name, cfg_path, resume_ckpt=None):
    cfg = Opts(cfg=cfg_path).parse_args([])
    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
    )
    print(model.device)
    return
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_steps=10,
        max_epochs=2,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=32,
        fast_dev_run=False if resume_ckpt is None else True,
        callbacks=[checkpoint_callback] if resume_ckpt is None else [],
        enable_checkpointing=True if resume_ckpt is None else False,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)
    del trainer
    del cfg
    del model
    del checkpoint_callback


def test_trainer(model_name):
    cfg_path = "tests/uts/default.yml"
    assert Path(cfg_path).exists(), "config file not found"
    print(cfg_path)

    train(model_name, cfg_path)
    train(
        model_name,
        cfg_path,
        resume_ckpt="lightning_logs/version_0/checkpoints/last.ckpt",
    )
# class QueryDataModule(pl.LightningDataModule):
#     """
#     texts
#     """
#     def __init__(self, cfg, batch_size):
#         super().__init__()
#         self.cfg = cfg
#         self.namee = 'aaaa'
#         self.batch_size = batch_size
#     def setup(self, stage: str):
#         self.prediction_ds = AIC22TextJsonDataset(**self.cfg.data['text'])
#     def predict_dataloader(self):
#         return DataLoader(self.prediction_ds, batch_size=self.batch_size, collate_fn=self.prediction_ds.collate_fn)

# class GalleryDataModule(pl.LightningDataModule):
#     """
#     images
#     """
#     def __init__(self, cfg, batch_size):
#         super().__init__()
#         self.cfg = cfg
#         self.namee = 'bbb'
#         self.batch_size = batch_size
#     def setup(self, stage: str):
#         self.prediction_ds = AIC22TrackJsonWithMotionDataset(**self.cfg.data['track'])
#     def predict_dataloader(self):
#         return DataLoader(self.prediction_ds, batch_size=self.batch_size, collate_fn=self.prediction_ds.collate_fn)


# def infer(model_name):
#     resume_ckpt="./lightning_logs/version_0/checkpoints/last.ckpt",
#     cfg_path = "tests/uts/default.yml"

#     cfg = Opts(cfg=cfg_path).parse_args([])
    
#     # TextDM = QueryDataModule(cfg, batch_size=1)
#     # TrackDM = GalleryDataModule(cfg, batch_size=1)
#     model = MODEL_REGISTRY.get(model_name)(cfg)
#     checkpoint_callback = ModelCheckpoint(
#         verbose=True,
#         save_last=True,
#     )
#     trainer = pl.Trainer(
#         log_every_n_steps=1,
#         max_steps=10,
#         max_epochs=2,
#         gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
#         check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
#         accelerator="ddp" if torch.cuda.device_count() > 1 else None,
#         sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
#         precision=32,
#         fast_dev_run=True,
#         callbacks=[checkpoint_callback],
#         enable_checkpointing=True,
#     )
#     trainer.fit(model, ckpt_path=resume_ckpt)

    # trainer.predict(model, TextDM)

def infer(model_name):

    resume_ckpt="./lightning_logs/version_1/checkpoints/last.ckpt",
    cfg_path = "tests/uts/default.yml"

    cfg = Opts(cfg=cfg_path).parse_args([])
    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
    )
    print(model.device)
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_steps=10,
        max_epochs=2,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=32,
        fast_dev_run=False if resume_ckpt is None else True,
        callbacks=[checkpoint_callback] if resume_ckpt is None else [],
        enable_checkpointing=True if resume_ckpt is None else False,
    )
    trainer.fit(model, ckpt_path=resume_ckpt)
    del trainer
    del cfg
    del model
    del checkpoint_callback
if __name__ == "__main__":
    test_trainer("UTS")
    infer("UTS")