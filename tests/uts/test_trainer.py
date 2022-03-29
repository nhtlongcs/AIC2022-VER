# python tests/trainer.py

import torch
from opt import Opts

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import MODEL_REGISTRY

from pathlib import Path
import pytest


@pytest.mark.parametrize("model_name", ["UTS"])
def test_trainer(tmp_path, model_name):
    cfg_path = "tests/uts/default.yml"
    assert Path(cfg_path).exists(), "config file not found"
    cfg = Opts(cfg=cfg_path).parse_args([])

    model = MODEL_REGISTRY.get(model_name)(cfg)
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_last=True,
    )
    
    trainer = pl.Trainer(
        default_root_dir="./runs",
        log_every_n_steps=1,
        max_steps=1,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        enable_checkpointing=True,
        fast_dev_run=False,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)

    # Resume
    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_steps=1,
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=cfg.trainer["evaluate_interval"],
        enable_checkpointing=False,
        fast_dev_run=True,
    )

    trainer.fit(
        model, 
        ckpt_path="./runs/lightning_logs/version_0/checkpoints/last.ckpt")
