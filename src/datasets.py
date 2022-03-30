import json
import os
import random

import torch
from PIL import Image
from registry import Registry
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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
        flip_aug=False,
        txt_classnames=None,
        **kwargs
    ):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg
        self.crop_area = data_cfg["CROP_AREA"]
        self.random = Random
        self.txt_classnames = txt_classnames
        if txt_classnames is not None:
            self._load_classnames()

        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transform = transform
        self.bk_dic = {}
        self.bk_cache = mo_cache
        self.tokenizer = AutoTokenizer.from_pretrained(tok_model_name)

        self.all_indexs = [self.classes_idx[i] for i in self.list_of_uuids]
        self.flip_tag = [False] * len(self.list_of_uuids)
        if flip_aug:
            for uuid, description in zip(self.list_of_uuids, self.list_of_tracks):
                texts = description["nl"]
                for j in range(len(texts)):
                    nl = texts[j]
                    if "turn" in nl:
                        if "left" in nl:
                            self.all_indexs.append(self.classes_idx[uuid])
                            self.flip_tag.append(True)
                            break
                        elif "right" in nl:
                            self.all_indexs.append(self.classes_idx[uuid])
                            self.flip_tag.append(True)
                            break
        print(len(self.all_indexs))
        print("data load")
    
    def _load_classnames(self):
        """
        Read data from csv and load into memory
        """
       
        with open(self.txt_classnames, 'r') as f:
            self.classnames = f.read().splitlines()
        
        self.classes_idx = {}
        # Mapping between classnames and indices
        for idx, classname in enumerate(self.classnames):
            self.classes_idx[classname] = idx
        self.num_classes = len(self.classnames)

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

