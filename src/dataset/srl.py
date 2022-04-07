import json
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
from . import DATASET_REGISTRY, default_loader


@DATASET_REGISTRY.register()
class CropVehDataset(Dataset):
    def __init__(
        self, data_cfg, json_path, color_csv, vehtype_csv, transform=None, **kwargs
    ):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg
        self.crop_area = data_cfg["CROP_AREA"]
        with open(json_path) as f:
            tracks = json.load(f)
        self.list_of_uuids = sorted(list(tracks.keys()))
        self.list_of_tracks = [tracks[i] for i in self.list_of_uuids]
        self.transform = transform
        self.all_indexs = list(range(len(self.list_of_uuids)))

        # === srl ===
        self.color_df = pd.read_csv(color_csv)[["query_id", "labels"]]
        self.vehtype_df = pd.read_csv(vehtype_csv)[["query_id", "labels"]]
        # convert labels column from string to list
        self.color_df["labels"] = self.color_df["labels"].apply(lambda x: eval(x))
        self.vehtype_df["labels"] = self.vehtype_df["labels"].apply(lambda x: eval(x))
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

        color_lbl = self.color_df[self.color_df["query_id"] == track_id][
            "labels"
        ].values[0]
        vehtype_lbl = self.vehtype_df[self.vehtype_df["query_id"] == track_id][
            "labels"
        ].values[0]

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

        return {
            "crop": crop,
            "color_lbl": torch.tensor(color_lbl),
            "vehtype_lbl": torch.tensor(vehtype_lbl),
        }

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x["crop"] for x in batch]),
            "color_lbls": torch.stack([x["color_lbl"] for x in batch]),
            "vehtype_lbls": torch.stack([x["vehtype_lbl"] for x in batch]),
        }
        return batch_dict

@DATASET_REGISTRY.register()
class AIC22TrackVehJsonDataset(Dataset):
    """
    """
    def __init__(
        self, 
        image_dir: str,
        json_path: str, 
        crop_area: float = 1.0, 
        transform=None, 
        **kwargs
    ):        
        super().__init__()
        self.json_path = json_path
        self.transform = transform
        self.image_dir = image_dir
        self.crop_area = crop_area
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

        return_dict = {
            'id': track_id,
            'crop': crop,
        }
        return return_dict

    def collate_fn(self, batch):
        crops = torch.stack([s['crop'] for s in batch])
        track_ids = [s['id'] for s in batch]

        return {
            'ids': track_ids,
            'images': crops,
        }