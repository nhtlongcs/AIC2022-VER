import pytorch_lightning as pl
import torch
from src.datasets import DATASET_REGISTRY
from src.metrics import METRIC_REGISTRY
from . import MODEL_REGISTRY
import torchvision

from src.utils.device import detach
from torch.utils.data import DataLoader


@MODEL_REGISTRY.register()
class AICBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.init_model()
        self.metric = {
            mcfg["name"]: METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
            for mcfg in config["metric"]
        }

    def init_model(self):
        raise NotImplementedError

    def prepare_data(self):
        image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((288, 288)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
            **self.cfg["data"]["args"]["train"],
            data_cfg=self.cfg["data"]["args"],
            tok_model_name=self.cfg["extractors"]["lang_encoder"]["args"]["pretrained"],
            transform=image_transform,
        )
        self.val_dataset = DATASET_REGISTRY.get(self.cfg["data"]["name"])(
            **self.cfg["data"]["args"]["val"],
            data_cfg=self.cfg["data"]["args"],
            tok_model_name=self.cfg["extractors"]["lang_encoder"]["args"]["pretrained"],
            transform=image_transform,
        )

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch).mean()
        # 3. Update monitor
        self.log("train/loss", detach(loss))

        return {"loss": loss, "log": {"train_loss": detach(loss)}}

    def compute_loss(self, visual_embeddings, nlang_embeddings, batch):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        # 1. Get embeddings from model
        output = self.forward(batch)
        # 2. Calculate loss
        loss = self.compute_loss(**output, batch=batch)
        # 3. Update metric for each batch
        for m in self.metric.values():
            m.update(output)

        return {"loss": detach(loss)}

    def validation_epoch_end(self, outputs):
        # 1. Calculate average validation loss
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # 2. Calculate metric value
        out = {"val_loss": loss}
        for k in self.metric.keys():
            self.metric[k].calculate()
            metric_dict = self.metric[k].value()
            out[k] = metric_dict["score"]
            self.metric[k].summary()
            self.log(f"val/{k}", out[k])
            for kk, vv in metric_dict["score_dict"].items():
                self.log(f"val/{k}/{kk}", vv)

        # 3. Reset metric
        for m in self.metric.values():
            m.reset()

        self.log("val/loss", loss.cpu().numpy().item())
        return {**out, "log": out}

    def train_dataloader(self):
        train_dataloader = DataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            **self.cfg["data"]["args"]["val"]["loader"],
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.trainer["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
