import json
import os
import random

import torch
import numpy as np
from PIL import Image
from registry import Registry
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List 
import numpy as np 
import os.path as osp
from src.extractors.clip.tokenization_clip import ClipTokenizer

from . import DATASET_REGISTRY, default_loader

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "true"  # https://github.com/huggingface/transformers/issues/5486




@DATASET_REGISTRY.register()
class CityFlowNLDataset(Dataset):
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
            }
        else:
            return {
                'track_id': track_id,
                'crop': crop,
                'text': text,
                'instance_id': torch.tensor(tmp_index),
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
            }
        else:
            batch_dict = {
                "images": torch.stack([x['crop'] for x in batch]),
                "texts": [x['text'] for x in batch],
                "car_ids": torch.stack([x['instance_id'] for x in batch]),
                "query_ids": [x['track_id'] for x in batch],
                "gallery_ids": [x['track_id'] for x in batch],
                "target_ids": [x['track_id'] for x in batch],
            }

        batch_dict["tokens"] = self.tokenizer.batch_encode_plus(
            batch_dict["texts"], padding="longest", return_tensors="pt"
        )
        return batch_dict

SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

@DATASET_REGISTRY.register()
class CityFlowNLClipDataset(Dataset):
    def __init__(
        self,
        data_cfg,
        json_path,
        transform=None,
        flip_aug: bool = False,
        num_texts_used: int = 3, 
        use_other_views: bool = False,
        max_words: int = 30,
        tok_model_name='bpe',
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
        self.max_words = max_words

        self.num_texts_used = num_texts_used
        self.use_other_views = use_other_views

        self.tokenizer = ClipTokenizer(tok_model_name)
        
        print(len(self.list_of_uuids))
        print("data load")

    def __len__(self):
        return len(self.list_of_uuids)

    def get_text(self, caption):
        # tokenize word
        words = self.tokenizer.tokenize(caption)

        # add cls token
        words = [SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        # add end token
        words = words + [SPECIAL_TOKEN["SEP_TOKEN"]]

        # convert token to id according to the vocab
        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        # add zeros for feature of the same length
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # ensure the length of feature to be equal with max words
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(segment_ids) == self.max_words
        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_segment = np.array(segment_ids)

        return (
          torch.from_numpy(pairs_text), 
          torch.from_numpy(pairs_mask), 
          torch.from_numpy(pairs_segment)
        )

    def __getitem__(self, index):

        track_id = self.list_of_uuids[index]
        track = self.list_of_tracks[index]
        frame_idx = int(random.uniform(0, len(track["frames"])))
        all_texts = track["nl"]
        if self.use_other_views:
            all_texts += track['nl_other_views']
        random.shuffle(all_texts)
        texts = np.random.choice(all_texts, size=min(len(all_texts), self.num_texts_used), replace=False)
        text = '. '.join(texts)

        is_flip = False
        if self.flip_aug:
            is_flip = np.random.rand() < 0.5
            if is_flip:
                text = (
                    text.replace("left", "888888")
                    .replace("right", "left")
                    .replace("888888", "right")
                )

        # obtain text data
        pairs_text, pairs_mask, pairs_segment = self.get_text(text)

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
            if self.list_of_uuids[index] in self.bk_dic:
                bk = self.bk_dic[self.list_of_uuids[index]]
            else:
                bk = default_loader(
                    self.data_cfg["MOTION_PATH"]
                    + "/%s.jpg" % self.list_of_uuids[index]
                )

            bk = self.transform(bk)
            if self.flip_aug and is_flip:
                bk = torch.flip(bk, [2])


        if self.data_cfg["USE_MOTION"]:
            return {
                'track_id': track_id,
                'text': text,
                'crop': crop,
                'encoded_text': [pairs_text, pairs_mask, pairs_segment],
                'instance_id': torch.tensor(index),
                'motion': bk,
            }
        else:
            return {
                'track_id': track_id,
                'text': text,
                'crop': crop,
                'encoded_text': [pairs_text, pairs_mask, pairs_segment],
                'instance_id': torch.tensor(index),
            }

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x['crop'] for x in batch]),
            "tokens": {
              "input_ids": torch.stack([x['encoded_text'][0] for x in batch]),
            },
            "texts": [x['text'] for x in batch],
            "car_ids": torch.stack([x['instance_id'] for x in batch]),
            "query_ids": [x['track_id'] for x in batch],
            "gallery_ids": [x['track_id'] for x in batch],
            "target_ids": [x['track_id'] for x in batch],
        }
        if self.data_cfg["USE_MOTION"]:
            batch_dict.update({
              "motions": torch.stack([x['motion'] for x in batch])
            })

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
        