from typing import List
import os.path as osp
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

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

        token_dict.update({
            'ids': text_ids
        })

        return token_dict


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