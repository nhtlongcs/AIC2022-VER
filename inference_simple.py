import json
import os.path as osp

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opt import Opts
from src.datasets import AIC22TextJsonDataset, AIC22TrackJsonWithMotionDataset
from src.models import MODEL_REGISTRY
from src.utils.device import move_to
from src.utils.faiss_retrieval import FaissRetrieval


@torch.no_grad()
def inference(model, device, lang_ds, visual_ds, top_k: int, savedir: str):
    
    
    lang_dataloader = DataLoader(lang_ds, batch_size=1, shuffle=False, collate_fn=lang_ds.collate_fn)
    visual_dataloader = DataLoader(visual_ds, batch_size=1, shuffle=False,collate_fn=visual_ds.collate_fn)
    model.eval()
    model = model.to(device)
    # Extract lang feats
    lang_results = {}
    for idx, batch in tqdm(enumerate(lang_dataloader)):
        batch = move_to(batch, device)
        lang_feats = model.encode_nlang_feats(batch)
        lang_feats = move_to(lang_feats, torch.device('cpu')).detach().numpy()
        ids = batch['ids']
        for lang_id, lang_feat in zip(ids, lang_feats):
            lang_results[lang_id] = lang_feat.tolist()
        
    with open(osp.join(savedir, 'text_embeds.json'), 'w') as f:
        json.dump(lang_results, f)

    # Extract visual feats
    visual_results = {}
    for idx, batch in tqdm(enumerate(visual_dataloader)):
        batch = move_to(batch, device)
        visual_feats, motion_feats = model.encode_visual_feats(batch)
        visual_feats = move_to(visual_feats, torch.device('cpu'))
        visual_feats = visual_feats.detach().numpy()
        ids = batch['ids']
        for visual_id, visual_feat in zip(ids, visual_feats):
            visual_results[visual_id] = visual_feat.tolist()
        
    with open(osp.join(savedir, 'visual_embeds.json'), 'w') as f:
        json.dump(visual_results, f)

    # Faiss retrieval
    retriever = FaissRetrieval(dimension=model.cfg.model['args']['EMBED_DIM'])
    query_embeddings = np.stack(lang_results.values(), axis=0).astype(np.float32)
    gallery_embeddings = np.stack(visual_results.values(), axis=0).astype(np.float32)
    query_ids = list(lang_results.keys())
    gallery_ids = list(visual_results.keys())

    retriever.similarity_search(
        query_embeddings,
        gallery_embeddings,
        query_ids,
        gallery_ids,
        top_k=top_k,
        save_results=osp.join(savedir, 'retrieval_results.json')
    )

if __name__ == "__main__":

    resume_ckpt = "./lightning_logs/version_0/checkpoints/last.ckpt"
    cfg_path = "test.yml"

    cfg = Opts(cfg=cfg_path).parse_args()
    model = MODEL_REGISTRY.get(cfg.model['name'])(cfg)
    model.load_from_checkpoint(resume_ckpt, config=cfg)
    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((288, 288)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    inference(
        model=model,
        device=torch.device('cuda:0'),
        lang_ds=AIC22TextJsonDataset(**cfg.data['text']),
        visual_ds=AIC22TrackJsonWithMotionDataset(**cfg.data['track'], transform=image_transform),
        top_k=5,
        savedir='./'
    )
