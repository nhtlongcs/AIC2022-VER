import json
import os
import random
import pandas as pd
from pyparsing import col 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List 
import numpy as np 
import os.path as osp

from . import DATASET_REGISTRY, default_loader

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "true"  # https://github.com/huggingface/transformers/issues/5486




@DATASET_REGISTRY.register()
class CityFlowSRLDataset(Dataset):
    def __init__(
        self,
        data_cfg,
        json_path,
        tok_model_name,
        transform=None,
        mo_cache=False,
        flip_aug: bool = False,
        num_texts_used: int = 3, 
        use_other_views: bool = False,
        **kwargs
    ):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg
        self.crop_area = data_cfg["CROP_AREA"]
        self.flip_aug = flip_aug
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = sorted(list(tracks.keys()))
        self.list_of_tracks = [tracks[i] for i in self.list_of_uuids]
        self.transform = transform
        self.bk_dic = {}
        self.bk_cache = mo_cache
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model_name)

        self.all_indexs = list(range(len(self.list_of_uuids)))
        self.flip_tag = [False] * len(self.list_of_uuids)

        self.num_texts_used = num_texts_used
        self.use_other_views = use_other_views
        
        # === srl ===
        self.color_df = pd.read_csv(data_cfg["COLOR_CSV"])[['query_id', 'labels']]
        self.vehtype_df = pd.read_csv(data_cfg["VEH_CSV"])[['query_id', 'labels']]
        # convert labels column from string to list 
        self.color_df['labels'] = self.color_df['labels'].apply(lambda x: eval(x))
        self.vehtype_df['labels'] = self.vehtype_df['labels'].apply(lambda x: eval(x))
        # === srl ===
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):
        tmp_index = self.all_indexs[index]
        track_id = self.list_of_uuids[tmp_index]
        track = self.list_of_tracks[tmp_index]
        frame_idx = int(random.uniform(0, len(track["frames"])))
        all_texts = track["nl"]
        if self.use_other_views:
            all_texts += track['nl_other_views']
        random.shuffle(all_texts)
        texts = np.random.choice(all_texts, size=min(len(all_texts), self.num_texts_used), replace=False)
        text = '. '.join(texts)

        color_lbl = self.color_df[self.color_df['query_id'] == track_id]['labels'].values[0]
        vehtype_lbl = self.vehtype_df[self.vehtype_df['query_id'] == track_id]['labels'].values[0]
        
        is_flip = False
        if self.flip_aug:
            is_flip = np.random.rand() < 0.5
            if is_flip:
                text = (
                    text.replace("left", "888888")
                    .replace("right", "left")
                    .replace("888888", "right")
                )


        frame_path = os.path.join(
            self.data_cfg["CITYFLOW_PATH"], track["frames"][frame_idx]
        )

        frame = default_loader(frame_path)
        box = track["boxes"][frame_idx]
        if self.crop_area == 1.6666667:
            box = (
                int(box[0] - box[2] / 3.0),
                int(box[1] - box[3] / 3.0),
                int(box[0] + 4 * box[2] / 3.0),
                int(box[1] + 4 * box[3] / 3.0),
            )
        else:
            box = (
                int(box[0] - (self.crop_area - 1) * box[2] / 2.0),
                int(box[1] - (self.crop_area - 1) * box[3] / 2),
                int(box[0] + (self.crop_area + 1) * box[2] / 2.0),
                int(box[1] + (self.crop_area + 1) * box[3] / 2.0),
            )

        crop = frame.crop(box)
        if self.transform is not None:
            crop = self.transform(crop)
            if self.flip_aug and is_flip:
                crop = torch.flip(crop, [2])   

        if self.data_cfg["USE_MOTION"]:
            if self.list_of_uuids[tmp_index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[tmp_index]]
            else:
                bk = default_loader(
                    self.data_cfg["MOTION_PATH"]
                    + "/%s.jpg" % self.list_of_uuids[tmp_index]
                )
            if self.bk_cache:
                self.bk_dic[self.list_of_uuids[tmp_index]] = bk

            bk = self.transform(bk)
            if self.flip_aug and is_flip:
                bk = torch.flip(bk, [2])


        if self.data_cfg["USE_MOTION"]:
            return {
                'track_id': track_id,
                'crop': crop,
                'text': text,
                'instance_id': torch.tensor(tmp_index),
                'motion': bk,
                'color_lbl': torch.tensor(color_lbl),
                'vehtype_lbl': torch.tensor(vehtype_lbl),
            }
        else:
            return {
                'track_id': track_id,
                'crop': crop,
                'text': text,
                'instance_id': torch.tensor(tmp_index),
                'color_lbl': torch.tensor(color_lbl),
                'vehtype_lbl': torch.tensor(vehtype_lbl),
            }

    def collate_fn(self, batch):
        if self.data_cfg["USE_MOTION"]:
            batch_dict = {
                "images": torch.stack([x['crop'] for x in batch]),
                "texts": [x['text'] for x in batch],
                "motions": torch.stack([x['motion'] for x in batch]),
                "car_ids": torch.stack([x['instance_id'] for x in batch]),
                "query_ids": [x['track_id'] for x in batch],
                "gallery_ids": [x['track_id'] for x in batch],
                "target_ids": [x['track_id'] for x in batch],
                "color_lbls": torch.stack([x['color_lbl'] for x in batch]),
                "vehtype_lbls": torch.stack([x['vehtype_lbl'] for x in batch]),
            }
        else:
            batch_dict = {
                "images": torch.stack([x['crop'] for x in batch]),
                "texts": [x['text'] for x in batch],
                "car_ids": torch.stack([x['instance_id'] for x in batch]),
                "query_ids": [x['track_id'] for x in batch],
                "gallery_ids": [x['track_id'] for x in batch],
                "target_ids": [x['track_id'] for x in batch],
                "color_lbls": torch.stack([x['color_lbl'] for x in batch]),
                "vehtype_lbls": torch.stack([x['vehtype_lbl'] for x in batch]),

            }

        batch_dict["tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["texts"], padding="longest", return_tensors="pt"
        )
        return batch_dict

