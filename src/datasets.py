import json
import os
import random

import torch
from PIL import Image
from registry import Registry
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List 
import numpy as np 
import os.path as osp

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "true"  # https://github.com/huggingface/transformers/issues/5486

DATASET_REGISTRY = Registry("DATASET")


def default_loader(path):
    return Image.open(path).convert("RGB")


@DATASET_REGISTRY.register()
class CityFlowNLDataset(Dataset):
    def __init__(
        self,
        data_cfg,
        json_path,
        tok_model_name,
        transform=None,
        Random=True,
        mo_cache=False,
        **kwargs
    ):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg
        self.crop_area = data_cfg["CROP_AREA"]
        self.random = Random
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
        flip_aug = False
        if flip_aug:
            for i in range(len(self.list_of_uuids)):
                text = self.list_of_tracks[i]["nl"]
                for j in range(len(text)):
                    nl = text[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(i)
                            self.flip_tag.append(True)
                            break
        print(len(self.all_indexs))
        print("data load")

    def __len__(self):
        return len(self.all_indexs)

    def __getitem__(self, index):

        tmp_index = self.all_indexs[index]
        flag = self.flip_tag[index]
        track = self.list_of_tracks[tmp_index]
        if self.random:
            nl_idx = int(random.uniform(0, 3))
            frame_idx = int(random.uniform(0, len(track["frames"])))
        else:
            nl_idx = 2
            frame_idx = 0
        text = track["nl"][nl_idx]
        if flag:
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

            if flag:
                crop = torch.flip(crop, [1])
                bk = torch.flip(bk, [1])

            return crop, text, bk, torch.tensor(tmp_index)
        if flag:
            crop = torch.flip(crop, [1])
        return crop, text, torch.tensor(tmp_index)

    def collate_fn(self, batch):
        if self.data_cfg["USE_MOTION"]:
            batch_dict = {
                "images": torch.stack([x[0] for x in batch]),
                "texts": [x[1] for x in batch],
                "motions": torch.stack([x[2] for x in batch]),
                "car_ids": torch.stack([x[3] for x in batch]),
            }
        else:
            batch_dict = {
                "images": torch.stack([x[0] for x in batch]),
                "texts": [x[1] for x in batch],
                "car_ids": torch.stack([x[2] for x in batch]),
            }

        batch_dict["tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["texts"], padding="longest", return_tensors="pt"
        )
        return batch_dict



class AIC22TextJsonDataset(Dataset):
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

        query_texts = np.random.choice(query_texts)
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


class AIC22TrackJsonWithMotionDataset(Dataset):
    """
    """
    def __init__(
        self, 
        image_dir: str,
        json_path: str, 
        crop_area: float = 1.0, 
        motion_path: str = None,
        transform=None, 
        meta_json: str = None,
        **kwargs
    ):        
        super().__init__()
        self.json_path = json_path
        self.transform = transform
        self.image_dir = image_dir
        self.crop_area = crop_area
        self.motion_path = motion_path
        self.meta_json = meta_json
        self._load_data()

    def _load_data(self):
        with open(self.json_path, 'r') as f:
            self.track_data = json.load(f)
        self.track_ids = list(self.track_data.keys())

    def __len__(self):
        return len(self.track_ids)

    def _load_meta(self):
        pass

    def __getitem__(self, index: int):
        track_id = self.track_ids[index]
        frame_names = self.track_data[track_id]['frames']
        boxes = self.track_data[track_id]['boxes']

        # Cropped instance image
        frame_idx = 0
        frame_path = osp.join(
            self.image_dir, frame_names[frame_idx]
        )
        frame = Image.open(frame_path).convert('RGB')
        box = boxes[frame_idx]
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
        if self.motion_path is not None:
            motion_image_path = osp.join(
                self.motion_path, track_id + '.jpg'
            )
            bk = Image.open(motion_image_path).convert('RGB')
            bk = self.transform(bk)

        return_dict = {
            'id': track_id,
            'crop': crop,
            'motion': bk
        }

        if self.meta_json:
            meta_info = self.meta_dict[track_id]
            return_dict.update(meta_info)

        return return_dict

    def collate_fn(self, batch: List):
        crops = torch.stack([s['crop'] for s in batch])
        motions = torch.stack([s['motion'] for s in batch])
        track_ids = [s['id'] for s in batch]

        return {
            'ids': track_ids,
            'images': crops,
            'motions': motions
        }
        