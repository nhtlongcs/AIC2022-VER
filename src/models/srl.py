import torch.nn as nn
import torch.nn.functional as F
from src.extractors import EXTRCT_REGISTRY

from . import MODEL_REGISTRY
from .abstract import ClsBase
from src.utils.losses import FocalLoss


class ClassifyBlock(nn.Module):
    def __init__(self, inp_dim, num_cls, embed_dim=1024) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inp_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.projection = nn.Linear(embed_dim, num_cls)

    def forward(self, x, return_embed=False):
        x = self.model(x)
        if return_embed:
            return x, self.projection(x)
        return self.projection(x)


@MODEL_REGISTRY.register()
class MultiLabelClassifer(ClsBase):
    def __init__(self, config):
        super().__init__(config)

    def init_model(self):
        embed_dim = self.cfg.model["args"]["EMBED_DIM"]

        self.visualExtrct = EXTRCT_REGISTRY.get(
            self.cfg.extractors["img_encoder"]["name"]
        )(**self.cfg.extractors["img_encoder"]["args"])

        self.img_in_dim = self.visualExtrct.feature_dim

        self.logits = ClassifyBlock(
            inp_dim=self.img_in_dim,
            num_cls=self.cfg.model["args"]["NUM_CLASS"],
            embed_dim=embed_dim,
        )

        self.loss = FocalLoss(num_classes=self.cfg.model["args"]["NUM_CLASS"])

    def normalize_head(
        self, embedding,
    ):
        return F.normalize(embedding, p=2, dim=-1)

    def forward(self, batch):
        assert "images" in batch.keys(), "Batch must contain images"
        embedding = self.visualExtrct(batch["images"])
        embedding = self.normalize_head(embedding)
        logits = self.logits(embedding, return_embed=False)
        return {"logits": logits}

    def compute_loss(self, logits, batch, **kwargs):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        assert "images" in batch.keys(), "Batch must contain images"
        preds = self.forward(batch)
        preds.update({'ids': batch['ids']})
        return preds

@MODEL_REGISTRY.register()
class VehColorClassifer(MultiLabelClassifer):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["color_lbls"].long())


@MODEL_REGISTRY.register()
class VehTypeClassifer(MultiLabelClassifer):
    def __init__(self, config):
        super().__init__(config)

    def compute_loss(self, logits, batch, **kwargs):
        return self.loss(logits, batch["vehtype_lbls"].long())
