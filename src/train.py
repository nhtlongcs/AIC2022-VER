import torch
from opt import Opts

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer import seed_everything

from src.models import MODEL_REGISTRY
from src.utils.path import prepare_checkpoint_path
from src.callbacks.visualizer_callbacks import VisualizerCallback

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
        filename=str(config["model"]["name"])
        + "{epoch}-{train_loss:.3f}-{val/Accurracy:.2f}",
        monitor="val/Accuracy",
        verbose=config["global"]["verbose"],
        save_top_k=3,
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/Accurracy",
        min_delta=0.0001,
        patience=15,
        verbose=False,
        mode="max",
    )

    Wlogger = WandbLogger(
        project="aic",
        name=train_id,
        save_dir=config["global"]["save_dir"],
        log_model="all",
        entity=config['global']['username']
    )

    trainer = pl.Trainer(
        max_epochs=config.trainer["num_epochs"],
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        check_val_every_n_epoch=config.trainer["evaluate_interval"],
        enable_checkpointing=[checkpoint_callback, early_stop_callback],
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        precision=16 if config["global"]["use_fp16"] else 32,
        fast_dev_run=config["global"]["debug"],
        logger=Wlogger,
        callbacks=[VisualizerCallback(
          motion_path = "/content/AIC2022-VER/data/meta/motion_map",
          gt_json_path = "/content/AIC2022-VER/data/meta/val.json",
          query_results_json = "temps/query_results.json",
          mapping_json = "temps/track_id_mapping.json"
        )],
        num_sanity_val_steps=-1
        # auto_lr_find=True,
    )
    trainer.fit(model)


if __name__ == "__main__":
    cfg = Opts(cfg="configs/template.yml").parse_args()
    seed_everything(seed=cfg["global"]["SEED"])
    train(cfg)
