import numpy as np
import json
from tqdm import tqdm
import os, json, cv2, random, sys
import os.path as osp
from glob import glob

import torch

# from models.extractor import Extractor
from srl_handler.utils.cfg import get_default_cfg
from srl_handler.utils.common import scan_images, dump_json
from models.centernet import CenternetExtractor


def get_centernet2_model():
    backbone_name = "Centernet2"
    cfg_file = (
        "models/CenterNet2/configs/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.yaml"
    )
    model_weight = "./weights/CenterNet2_R2-101-DCN-BiFPN_4x+4x_1560_ST.pth"
    model = CenternetExtractor(cfg_file, model_weight, threshold=0.5)
    model.set_eval()
    print("Create model successfully")

    return model


def get_all_vehicles(img_dir, save_path, start_frame=0, end_frame=20, skip_frame=10):
    model = get_centernet2_model()
    VEHICLE_IDS = [
        # 1, #bicycle
        2,  # car
        # 3, #motorcycle
        5,  # bus
        6,  # train
        7,  # truck
    ]
    result = []
    no_boxes_frames = []
    qid = img_dir.split("/")[-1]
    if start_frame == 0 and end_frame == -1:
        list_frame_orders = [int(fname.split(".")[0]) for fname in os.listdir(img_dir)]
        start_frame, end_frame = min(list_frame_orders), max(list_frame_orders)

    list_fname = [
        "%06d.jpg" % count for count in range(start_frame, end_frame + 1, skip_frame)
    ]

    # for fname in tqdm(os.listdir(img_dir)):
    for fname in tqdm(list_fname):
        fpath = osp.join(img_dir, fname)
        cv_img = cv2.imread(fpath)

        preds, _ = model.inference_image(cv_img)
        list_ids = preds.pred_classes.cpu().numpy().tolist()
        scores = preds.scores.cpu().numpy()
        list_ids = [i in VEHICLE_IDS for i in list_ids]
        pred_boxes = preds.pred_boxes.tensor.cpu().numpy()  # xyxy
        pred_boxes = pred_boxes[list_ids]
        scores = scores[list_ids]

        # pred_boxes = pred_boxes[np.where(scores > 0.75)]
        result.append({fname: pred_boxes.tolist()})

    # save_path = osp.join(save_dir, f'{qid}.json')
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
        pass
    pass


if __name__ == "__main__":
    # video_names = ['video_4', 'video_5', 'video_6']
    video_names = ["video_7"]
    for vid_name in video_names:
        print(f"Run {vid_name}")
        img_dir = f"/content/AI_City_2021/dataset/vn_example/results/{vid_name}"
        save_dir = "/content/AI_City_2021/results/vn_example/det_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(
            save_dir, f"{vid_name}.json"
        )  #'/content/AI_City_2021/results/vn_example/video_30.json'
        get_all_vehicles(img_dir, save_path, start_frame=0, end_frame=-1, skip_frame=10)
        print(f"save result to {save_path}")
    pass
