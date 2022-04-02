import pytorch_lightning as pl
import torch
from src.datasets import DATASET_REGISTRY
from src.metrics import METRIC_REGISTRY
from . import MODEL_REGISTRY
import torchvision
from typing import Any
from src.utils.device import detach
from torch.utils.data import DataLoader
from src.metrics.metric_wrapper import RetrievalMetric
from src.utils.device import move_to

@MODEL_REGISTRY.register()
class AICBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        self.init_model()

    def init_model(self):
        raise NotImplementedError

    def setup(self, stage: str):
        if stage != 'predict':
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

            self.metric = RetrievalMetric(
                metrics = [
                    METRIC_REGISTRY.get(mcfg["name"])(**mcfg["args"])
                    if mcfg["args"] else METRIC_REGISTRY.get(mcfg["name"])()
                    for mcfg in self.cfg["metric"] 
                ], **self.cfg['metric_configs']
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
        self.metric.update(output, batch)

        return {"loss": detach(loss)}

    def validation_epoch_end(self, outputs):

        # 1. Calculate average validation loss
        loss = torch.mean(torch.stack([o["loss"] for o in outputs], dim=0))
        # 2. Calculate metric value
        out = {"val_loss": loss}

        # 3. Update metric for each batch
        metric_dict = self.metric.value()
        out.update(metric_dict)
        for k in metric_dict.keys():
            self.log(f"val/{k}", out[k])

        # Log string
        log_string = ""
        for metric, score in out.items():
            if isinstance(score, (int, float)):
                log_string += metric +': ' + f"{score:.5f}" +' | '
        log_string +='\n'
        print(log_string)

        # 4. Reset metric
        self.metric.reset()

        self.log("val/loss", loss.cpu().numpy().item())
        return {**out, "log": out}

    def train_dataloader(self):
        train_loader  = DataLoader(
            **self.cfg["data"]["args"]["train"]["loader"],
            dataset=self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
        )
        return train_loader 

    def val_dataloader(self):
        val_loader = DataLoader(
            **self.cfg["data"]["args"]["val"]["loader"],
            dataset=self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
        )
        return val_loader
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.cfg.trainer['lr'])
        
        train_set_len = len(self.train_dataset)
        train_bs = self.cfg["data"]["args"]["train"]["loader"]['batch_size']

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.trainer['lr'], 
                        steps_per_epoch=int(train_set_len//train_bs),
                        epochs=self.cfg.trainer['num_epochs'],
                        anneal_strategy='linear')

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        assert self.result_dir is None, "Result dir is not set"
        import pdb; pdb.set_trace()
        pred_batch = self.forward(batch)

        all_pairs = pred_batch['all_pairs']
        step_results= {}
        for i, (visual_embeds, lang_embeds) in enumerate(all_pairs):
            step_results[i] = {'visual_embeds': visual_embeds, 'lang_embeds': lang_embeds}
        
        return self.transfer_batch_to_device(step_results, torch.device('cpu'))

    def on_predict_end(self, outputs) -> None:
        assert len(outputs) >= 1, "No batch to predict"
        result = {}
        for i in range(outputs[0]):        
            visual_embeds = torch.stack([o["visual_embeds"] for o in outputs[i]], dim=0)
            lang_embeds = torch.stack([o["lang_embeds"] for o in outputs[i]], dim=0)
            result[i] = {'visual_embeds': visual_embeds, 'lang_embeds': lang_embeds}
        return result