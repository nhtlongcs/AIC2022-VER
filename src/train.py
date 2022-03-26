import torch
from opt import Opts

import pytorch_lightning as pl

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import seed_everything

from src.models import MODEL_REGISTRY
from src.utils.path import prepare_checkpoint_path


def train(config):
    model = MODEL_REGISTRY.get(config["model"]["name"])(config)
    pretrained_path = config["global"]["pretrained"]

    if pretrained_path:
        model = model.load_from_checkpoint(pretrained_path)

    cp_path, train_id = prepare_checkpoint_path(
        config["global"]["save_dir"], config["global"]["name"]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cp_path,
        filename="/{epoch}-{train_loss:.3f}-{Accurracy:.2f}",
        monitor="Accuracy",
        verbose=config["global"]["verbose"],
        save_top_k=3,
    )

    Wlogger = WandbLogger(
        project="aic",
        name=train_id,
        save_dir=config["global"]["save_dir"],
        log_model="all",
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer["num_epochs"],
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=config.trainer["evaluate_interval"],
        enable_checkpointing=[checkpoint_callback],
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16 if config["global"]["use_fp16"] else 32,
        fast_dev_run=config["global"]["debug"],
        logger=Wlogger,
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    train(cfg)
