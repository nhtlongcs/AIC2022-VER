"""
Script for visualizing tracking annotation from AIC22 dataset
"""

import json
import os
import os.path as osp
import cv2
import pandas as pd
from tqdm import tqdm
from utils.drawing import draw_one_box
from utils.bb_utils import xywh_to_xyxy_lst

from utils.constants import (
    TEST_CAM_IDS, TEST_TRACKS_JSON, TEST_AUX_TRACKS_JSON, TEST_TRACK_VIDEOS,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON, TRAIN_AUX_TRACKS_JSON, TRAIN_TRACK_VIDEOS,
    EXTRACTED_FRAMES_DIR
)

SPLIT = 'test' # or test
NEIGHBOR = True

if SPLIT == 'train':
    CAM_IDS = TRAIN_CAM_IDS
    if NEIGHBOR:
        TRACKS_JSON = TRAIN_TRACKS_JSON
    else:
        TRACKS_JSON = TRAIN_AUX_TRACKS_JSON
    OUTDIR = TRAIN_TRACK_VIDEOS
else:
    CAM_IDS = TEST_CAM_IDS
    if NEIGHBOR:
        TRACKS_JSON = TEST_TRACKS_JSON
    else:
        TRACKS_JSON = TEST_AUX_TRACKS_JSON
    OUTDIR = TEST_TRACK_VIDEOS

os.makedirs(OUTDIR, exist_ok=True)

def run():

    with open(TRACKS_JSON, 'r') as f:
        main_data = json.load(f)

    track_ids = list(main_data.keys())
    for track_id in tqdm(track_ids):
        frame_names = main_data[track_id]['frames']
        boxes = xywh_to_xyxy_lst(main_data[track_id]['boxes'])

        img = cv2.imread(
            osp.join(EXTRACTED_FRAMES_DIR, frame_names[0][2:])
        )
        height, width = img.shape[:-1]

        writer = cv2.VideoWriter(
            osp.join(OUTDIR, track_id+'.mp4'),   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            10, 
            (width, height))

        for frame_name, box in zip(frame_names, boxes):
            # Frame image
            img = cv2.imread(
                osp.join(EXTRACTED_FRAMES_DIR, frame_name[2:]))

            img = draw_one_box(
                img, 
                box, 
                key=f'id: {track_id}',
                color=[0,255,0])

            writer.write(img)

if __name__ == '__main__':
    run()