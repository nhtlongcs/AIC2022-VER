import json
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List 
import numpy as np 

from . import DATASET_REGISTRY, default_loader

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "true"  # https://github.com/huggingface/transformers/issues/5486




@DATASET_REGISTRY.register()
class CityFlowNLDatasetSubject(Dataset):
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


        # Subject texts, different idea from v1
        sub_text = '. '.join(track['subjects'] )

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
                'sub_text': sub_text
            }
        else:
            return {
                'track_id': track_id,
                'crop': crop,
                'text': text,
                'instance_id': torch.tensor(tmp_index),
                'sub_text': sub_text
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
                "sub_texts": [x['sub_text'] for x in batch]
            }
        else:
            batch_dict = {
                "images": torch.stack([x['crop'] for x in batch]),
                "texts": [x['text'] for x in batch],
                "car_ids": torch.stack([x['instance_id'] for x in batch]),
                "query_ids": [x['track_id'] for x in batch],
                "gallery_ids": [x['track_id'] for x in batch],
                "target_ids": [x['track_id'] for x in batch],
                "sub_texts": [x['sub_text'] for x in batch]
            }

        batch_dict["tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["texts"], padding="longest", return_tensors="pt"
        )

        batch_dict["sub_tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["sub_texts"], padding="longest", return_tensors="pt"
        )

        return batch_dict


class AIC22TextJsonDatasetSubject(Dataset):
    """
    """
    def __init__(
        self, 
        json_path: str, 
        tok_model_name: str = "bert-base-uncased", 
        num_texts_used: int = 3, 
        use_other_views: bool = False,
        **kwargs):

        super().__init__()

        self.json_path = json_path
        self.tok_model_name = tok_model_name
        self.num_texts_used = num_texts_used
        self.use_other_views = use_other_views
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model_name)
        self._load_data()

    def _load_data(self):
        with open(self.json_path, 'r') as f:
            self.queries_data = json.load(f)
        self.query_ids = list(self.queries_data.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, index: int):
        query_id = self.query_ids[index]
        query_data = self.queries_data[query_id]

        query_texts = query_data['nl']
        if self.use_other_views:
            query_texts += query_data['nl_other_views']

        query_texts = np.random.choice(query_texts, size=min(len(query_texts), self.num_texts_used), replace=False)
        query_text = '. '.join(query_texts)

        return {
            'id': query_id,
            'text': query_text
        }

    def collate_fn(self, batch: List):
        text_ids = [s['id'] for s in batch]
        texts =[s['text'] for s in batch]

        token_dict = self.tokenizer.batch_encode_plus(
            texts, padding="longest", return_tensors="pt"
        )
        batch_dict = {
            'ids': text_ids,
            'tokens': token_dict
        }
        return batch_dict
        # ```
        # token_dict.update({
        #     'ids': text_ids
        # }) 
        # return token_dict
        # ```
        # Above commented lines will cause error when move batch to device
        # because token_dict is <class 'transformers.tokenization_utils_base.BatchEncoding'>, already have to(device) method implemented
        # so if append new key to token_dict, it will cause error in self.to(device) (self is BatchEncoding)
        # Avoiding using move to device manually if it implemented themself

