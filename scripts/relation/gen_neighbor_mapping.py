"""
Script for generating all neighbor tracks based on annotation from AIC22 dataset
Read in the auxiliary tracks and the main tracks and decide which is the neighbor to which
"""

import json
import pandas as pd
from tqdm import tqdm
from external.relation.frame_utils import get_frame_ids_by_names, get_camera_id_by_name

from scripts.relation.constants import (
    AIC22_ORI_ROOT,
    TEST_CAM_IDS, TEST_TRACKS_JSON,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON,
    TEST_AUX_TRACKS_MAPPING_JSON,
    TRAIN_AUX_TRACKS_MAPPING_JSON
)

SPLIT = 'train' # or test
NUM_FRAMES_THRESHOLD = 5 # filter out tracks which appear less than threshold

if SPLIT == 'train':
    FOLDER_NAME = 'validation' #because AIC22 structure folder this way
    OUTPATH = TRAIN_AUX_TRACKS_MAPPING_JSON
    CAM_IDS = TRAIN_CAM_IDS
    TRACKS_JSON = TRAIN_TRACKS_JSON
else:
    FOLDER_NAME = 'train' #because AIC22 structure folder this way
    OUTPATH = TEST_AUX_TRACKS_MAPPING_JSON
    CAM_IDS = TEST_CAM_IDS
    TRACKS_JSON = TEST_TRACKS_JSON

ANNO = "{AIC22_ORI_ROOT}/{FOLDER_NAME}/{CAMERA}/gt/gt.txt"


def generate_neighbor_tracks_mapping(camera_id):

    df = pd.read_csv(ANNO.format(CAMERA=camera_id, FOLDER_NAME=FOLDER_NAME, AIC22_ORI_ROOT=AIC22_ORI_ROOT))
    df.columns = [
        'frame_id', 
        'track_id', 
        'x', 'y', 'w', 'h', 
        'conf', 'unk1', 'unk2', 'unk3'
    ]

    with open(TRACKS_JSON, 'r') as f:
        data = json.load(f)

    main_track_ids = list(data.keys())

    neighbor_mapping = {}
    for main_track_id in main_track_ids:

        frame_names = data[main_track_id]['frames']
        main_boxes = data[main_track_id]['boxes']

        # The camera id of the track
        current_camera_id = get_camera_id_by_name(frame_names[0])
        if current_camera_id != camera_id:
            continue

        # All the frames that main track appears
        frame_ids = get_frame_ids_by_names(frame_names)

        neighbor_mapping[main_track_id] = []
        # tracks that appear at same  frame with main track
        aux_appearances = {}
        for (frame_id, main_box) in zip(frame_ids, main_boxes):
            aux_df = df[df.frame_id == frame_id]
            for _, row in aux_df.iterrows():

                track_id, x, y, w, h = row[1:6]
                unique_track_id = f"{camera_id.replace('/', '_')}_{track_id}"

                main_box[0] += main_box[2]
                main_box[1] += main_box[3]
                other_box = [x,y,x+w,y+h]

                # Store the neighbor track candidates
                if unique_track_id not in aux_appearances.keys():
                    aux_appearances[unique_track_id] = []
                aux_appearances[unique_track_id].append(other_box)

        # filter out tracks which appear less than threshold
        aux_tracks = {k:v for k,v in aux_appearances.items() if len(v) >= NUM_FRAMES_THRESHOLD}
        
        for aux_track_id in aux_tracks.keys():
            neighbor_mapping[main_track_id].append(aux_track_id)

    return neighbor_mapping
        
def run():

    final_dict = {}
    for camera_id in tqdm(CAM_IDS):
        camera_neighbor_dict = generate_neighbor_tracks_mapping(camera_id)
        final_dict.update(camera_neighbor_dict)

    with open(OUTPATH, 'w') as f:
        json.dump(final_dict, f, indent=4)

if __name__ == '__main__':
    run()