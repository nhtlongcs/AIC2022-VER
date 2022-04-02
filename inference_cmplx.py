import torchvision
import torch
from opt import Opts

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.models import MODEL_REGISTRY
from src.datasets import AIC22TextJsonDataset, AIC22TrackJsonWithMotionDataset
import json 
import os.path as osp 
from src.utils.faiss_retrieval import FaissRetrieval
import numpy as np

class QueryDataModule(pl.LightningDataModule):
    """
    texts
    """
    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
    def setup(self, stage: str):
        self.prediction_ds = AIC22TextJsonDataset(**self.cfg.data['text'])
    def predict_dataloader(self):
        return DataLoader(self.prediction_ds, batch_size=self.batch_size, collate_fn=self.prediction_ds.collate_fn)

class GalleryDataModule(pl.LightningDataModule):
    """
    images
    """
    def __init__(self, cfg, batch_size):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((288, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    def setup(self, stage: str):
        self.prediction_ds = AIC22TrackJsonWithMotionDataset(**self.cfg.data['track'], transform=self.transform)
    def predict_dataloader(self):
        return DataLoader(self.prediction_ds, batch_size=self.batch_size, collate_fn=self.prediction_ds.collate_fn)

def pred2dict(preds, feat_key=None):
    results = {}
    for batch in preds:
        ids = batch['ids']
        feats = batch[feat_key].numpy()
        for id, feat in zip(ids, feats):
            results[id] = feat.tolist()
    return results

def inference(savedir='./', top_k=10):

    resume_ckpt = "./lightning_logs/version_0/checkpoints/last.ckpt"
    cfg_path = "test.yml"

    cfg = Opts(cfg=cfg_path).parse_args()
    model = MODEL_REGISTRY.get(cfg.model['name'])(cfg)
    model.load_from_checkpoint(resume_ckpt, config=cfg)
    print(model.device)
    TextDM = QueryDataModule(cfg, batch_size=1)
    VisualDM = GalleryDataModule(cfg, batch_size=1)

    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.device_count() else None,  # Use all gpus available
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        enable_checkpointing=False,
    )
    t_preds = trainer.predict(model, TextDM)
    t_preds = pred2dict(t_preds, feat_key='lang_embeds')
    i_preds = trainer.predict(model, VisualDM)
    i_preds = pred2dict(i_preds, feat_key='visual_embeds')
    with open(osp.join(savedir, 'text_embeds.json'), 'w') as f:
        json.dump(t_preds, f)
    with open(osp.join(savedir, 'visual_embeds.json'), 'w') as f:
        json.dump(i_preds, f)

    # Faiss retrieval
    retriever = FaissRetrieval(dimension=model.cfg.model['args']['EMBED_DIM'])
    query_embeddings = np.stack(t_preds.values(), axis=0).astype(np.float32)
    gallery_embeddings = np.stack(i_preds.values(), axis=0).astype(np.float32)
    query_ids = list(t_preds.keys())
    gallery_ids = list(i_preds.keys())

    retriever.similarity_search(
        query_embeddings,
        gallery_embeddings,
        query_ids,
        gallery_ids,
        top_k=top_k,
        save_results=osp.join(savedir, 'retrieval_results.json')
    )

    del trainer
    del cfg
    del model
    del TextDM
    del VisualDM
    del retriever
    del query_embeddings
    del gallery_embeddings


if __name__ == "__main__":
    inference()
