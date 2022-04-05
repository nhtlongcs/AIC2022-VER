"""
Script for visualizing tracking annotation from AIC22 dataset
"""

import json
import os
import os.path as osp
import cv2
import pandas as pd
from tqdm import tqdm
from utils.drawing import visualize_one_frame
from utils.bb_utils import bb_intersection_over_union

from utils.constants import (
    AIC22_ROOT,
    TEST_CAM_IDS, TEST_TRACKS_JSON,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON,
    TEST_AUX_TRACKS_MAPPING_JSON,
    TRAIN_AUX_TRACKS_MAPPING_JSON
)

CAMERA = "S01/c003"
FRAME_DIR = f"K:/Github/AIC2022-VER/data/meta/extracted_frames/train/{CAMERA}/img1"
ANNO = f"K:/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/train/{CAMERA}/gt/gt.txt"
OUTDIR = f"K:/Github/AIC2022-VER/data/meta/track_videos/neightbor_tracks/videos/{CAMERA}"
SPECIAL_TRACKS = "K:/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/test_tracks.json"
os.makedirs(OUTDIR, exist_ok=True)

def run():
    with open(SPECIAL_TRACKS, 'r') as f:
        data = json.load(f)

    df = pd.read_csv(ANNO)
    df.columns = [
        'frame_id', 
        'track_id', 
        'x', 'y', 'w', 'h', 
        'conf', 'unk1', 'unk2', 'unk3'
    ]

    track_ids = list(data.keys())
    for track_id in tqdm(track_ids):
        frame_names = data[track_id]['frames']
        boxes = data[track_id]['boxes']

        if CAMERA not in frame_names[0]:
            continue

        # "./train/S01/c003/img1/000028.jpg",
        frame_ids = [
            int(i.split('/')[-1][:-4])
            for i in frame_names
        ]
        
        img = cv2.imread(osp.join(FRAME_DIR, str(frame_ids[0]).zfill(6)+'.jpg'))
        height, width = img.shape[:-1]

        writer = cv2.VideoWriter(
            osp.join(OUTDIR, track_id+'.mp4'),   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            10, 
            (width, height))

        for frame_id, target_box in tqdm(zip(frame_ids, boxes)):
            
            target_box[2] += target_box[0]
            target_box[3] += target_box[1]

            # Frame image
            img = cv2.imread(osp.join(FRAME_DIR, str(frame_id).zfill(6)+'.jpg'))

            # Target object
            frame_dict = {
                'track_id': [-1], 
                'x1': [target_box[0]], 'y1': [target_box[1]], 
                'x2':  [target_box[2]], 'y2':  [target_box[3]],
                'color': [[0, 255, 0]]
            }

            # Other objects
            small_df = df[df.frame_id==frame_id]
            anns = [
                i for i in zip(
                    small_df.track_id, 
                    small_df.x, 
                    small_df.y, 
                    small_df.w, 
                    small_df.h)
            ]

            for (track_id, x, y, w, h) in anns:

                other_box = [x,y,x+w,y+h]
                if bb_intersection_over_union(other_box, target_box) > 0.5:  # filter out main tracks from other tracks
                    continue

                frame_dict['track_id'].append(track_id)
                frame_dict['x1'].append(other_box[0])
                frame_dict['y1'].append(other_box[1])
                frame_dict['x2'].append(other_box[2])
                frame_dict['y2'].append(other_box[3])
                frame_dict['color'].append([0,0,255])

            frame_df = pd.DataFrame(frame_dict)

            img = visualize_one_frame(img, frame_df)

            writer.write(img)



if __name__ == '__main__':
    run()