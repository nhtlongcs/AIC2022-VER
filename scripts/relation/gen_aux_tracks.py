"""
Script for generating all neighbor tracks based on annotation from AIC22 dataset
Read in tracking results and convert into same format as AIC22 tracks
"""

import json
import pandas as pd
from tqdm import tqdm
from scripts.relation.constants import (
    AIC22_ORI_ROOT,
    TEST_CAM_IDS, TEST_TRACKS_JSON,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON,
    TEST_AUX_TRACKS_JSON,
    TRAIN_AUX_TRACKS_JSON
)

SPLIT = 'train' # or test
NUM_FRAMES_THRESHOLD = 5 # filter out tracks which appear less than threshold

if SPLIT == 'train':
    FOLDER_NAME = 'validation' #because AIC22 structure folder this way
    OUTPATH = TRAIN_AUX_TRACKS_JSON
    CAM_IDS = TRAIN_CAM_IDS
    TRACKS_JSON = TRAIN_TRACKS_JSON
else:
    FOLDER_NAME = 'train' #because AIC22 structure folder this way
    OUTPATH = TEST_AUX_TRACKS_JSON
    CAM_IDS = TEST_CAM_IDS
    TRACKS_JSON = TEST_TRACKS_JSON

ANNO = "{AIC22_ORI_ROOT}/{FOLDER_NAME}/{CAMERA}/gt/gt.txt"

def generate_unique_neighbor_tracks(camera_id):
    df = pd.read_csv(ANNO.format(CAMERA=camera_id, FOLDER_NAME=FOLDER_NAME, AIC22_ORI_ROOT=AIC22_ORI_ROOT))
    df.columns = [
        'frame_id', 
        'track_id', 
        'x', 'y', 'w', 'h', 
        'conf', 'unk1', 'unk2', 'unk3'
    ]

    neighbor_dict = {}
    track_ids = list(df.track_id)
    for track_id in track_ids:
        track_df = df[df.track_id == track_id]

        unique_track_id = f"{camera_id.replace('/', '_')}_{track_id}"

        neighbor_dict[unique_track_id] = {
            'frames': [],
            'boxes': []
        }

        for _, row in track_df.iterrows():
            frame_id, _, x, y, w, h = row[:6]
            neighbor_dict[unique_track_id]['frames'].append(
                f"./{FOLDER_NAME}/{camera_id}/img1/{str(frame_id).zfill(6)}.jpg"
            )

            neighbor_dict[unique_track_id]['boxes'].append([x, y, w, h])

    return neighbor_dict
        
def run():

    final_dict = {}
    for camera_id in tqdm(CAM_IDS):
        camera_neighbor_dict = generate_unique_neighbor_tracks(camera_id)
        final_dict.update(camera_neighbor_dict)

    with open(OUTPATH, 'w') as f:
        json.dump(final_dict, f, indent=4)

if __name__ == '__main__':
    run()