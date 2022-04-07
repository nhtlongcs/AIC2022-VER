"""
Script for visualizing tracking annotation from AIC22 dataset
"""

import json
import os
import os.path as osp
import cv2
from tqdm import tqdm
import pandas as pd
from external.relation.drawing import visualize_one_frame
from external.relation.bb_utils import xywh_to_xyxy_lst
from external.relation.frame_utils import get_frame_ids_by_names

from scripts.relation.constants import (
    TEST_CAM_IDS, TEST_TRACKS_JSON, TEST_RELATION_JSON, TEST_AUX_TRACKS_JSON,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON, TRAIN_RELATION_JSON, TRAIN_AUX_TRACKS_JSON,
    TRAIN_TRACK_RELATION_VIDEOS, TEST_TRACK_RELATION_VIDEOS,
    EXTRACTED_FRAMES_DIR
)

SPLIT = 'test' # or test

if SPLIT == 'train':
    CAM_IDS = TRAIN_CAM_IDS
    TRACKS_JSON = TRAIN_TRACKS_JSON
    RELATION_TRACKS_JSON = TRAIN_RELATION_JSON
    AUX_TRACKS_JSON = TRAIN_AUX_TRACKS_JSON
    OUTDIR = TRAIN_TRACK_RELATION_VIDEOS
else:
    CAM_IDS = TEST_CAM_IDS
    TRACKS_JSON = TEST_TRACKS_JSON
    RELATION_TRACKS_JSON = TEST_RELATION_JSON
    AUX_TRACKS_JSON = TEST_AUX_TRACKS_JSON
    OUTDIR = TEST_TRACK_RELATION_VIDEOS

os.makedirs(OUTDIR, exist_ok=True)

def visualize_neighbors():

    with open(TEST_TRACKS_JSON, 'r') as f:
        main_data = json.load(f)

    with open(TEST_RELATION_JSON, 'r') as f:
        neighbor_mapping = json.load(f)

    with open(AUX_TRACKS_JSON, 'r') as f:
        aux_data = json.load(f)


    main_track_ids = list(neighbor_mapping.keys())

    for main_track_id in tqdm(main_track_ids):

        # Main track info
        main_boxes = xywh_to_xyxy_lst(main_data[main_track_id]['boxes'])
        main_frame_names = main_data[main_track_id]['frames']

        # Init video writer
        tmp_path = osp.join(EXTRACTED_FRAMES_DIR, main_frame_names[0][2:])
        img = cv2.imread(tmp_path)
        height, width = img.shape[:-1]

        if len(main_frame_names) > 10:
            fps = 10
        else:
            fps = len(main_frame_names)//2

        writer = cv2.VideoWriter(
            osp.join(OUTDIR, main_track_id+'.mp4'),   
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height))

        main_frame_ids = get_frame_ids_by_names(main_frame_names)
        main_frame_ids.sort()

        # Neighbor infos        
        neighbors = neighbor_mapping[main_track_id]
        followed_byed_ids = neighbors['followed_by']
        follow_ids = neighbors['follow']

        if len(followed_byed_ids) == 0 and len(follow_ids) == 0:
            continue

        # FOR EASIEST WAY, WE USE DATAFRAME TO STORE ALL TRACKS, THEN VISUALIZE THIS DATAFRAME #
        
        ## Create list of dicts to generate dataframe later
        track_lists = []

        # Visualize main track first
        for main_box, main_frame_id in zip(main_boxes, main_frame_ids):
            track_lists.append({
                'frame_id': main_frame_id,
                'track_id': -1, 
                'x1': main_box[0], 'y1': main_box[1], 
                'x2':  main_box[2], 'y2':  main_box[3],
                'color': [0, 255, 0]
            })


        if len(followed_byed_ids) > 0:
            # Visualize followed by
            followed_byed_neighbors = [(id, aux_data[id]) for id in followed_byed_ids]

            for neighbor_track_id, neighbor in followed_byed_neighbors:
                neighbor_boxes = xywh_to_xyxy_lst(neighbor['boxes'])
                neighbor_frames = neighbor['frames']
                neighbor_frames_ids = get_frame_ids_by_names(neighbor_frames)
                neighbor_frames_ids.sort()
                intersected = [
                    (box, id) for id, box in zip(neighbor_frames_ids, neighbor_boxes)
                    if id in main_frame_ids
                ]

                for neighbor_box, neighbor_frame_id in intersected:
                    track_lists.append({
                        'frame_id': neighbor_frame_id,
                        'track_id': neighbor_track_id + '- followed by', 
                        'x1': neighbor_box[0], 'y1': neighbor_box[1], 
                        'x2':  neighbor_box[2], 'y2':  neighbor_box[3],
                        'color': [0, 255, 255]
                    })

        if len(follow_ids) > 0:
            # Visualize following
            follow_neighbors = [(id, aux_data[id]) for id in follow_ids]

            for neighbor_track_id, neighbor in follow_neighbors:
                neighbor_boxes = xywh_to_xyxy_lst(neighbor['boxes'])
                neighbor_frames = neighbor['frames']
                neighbor_frames_ids = get_frame_ids_by_names(neighbor_frames)
                neighbor_frames_ids.sort()
                intersected = [
                    (box, id) for id, box in zip(neighbor_frames_ids, neighbor_boxes)
                    if id in main_frame_ids
                ]

                for neighbor_box, neighbor_frame_id in intersected:
                    track_lists.append({
                        'frame_id': neighbor_frame_id,
                        'track_id': neighbor_track_id + '- following', 
                        'x1': neighbor_box[0], 'y1': neighbor_box[1], 
                        'x2':  neighbor_box[2], 'y2':  neighbor_box[3],
                        'color': [0, 0, 255]
                    })


        # Write to video
        track_df = pd.DataFrame(track_lists)

        for frame_id, frame_name in zip(main_frame_ids, main_frame_names):
            # Frame image
            img = cv2.imread(
                osp.join(EXTRACTED_FRAMES_DIR, frame_name[2:]))

            # All tracks in that frame
            frame_df = track_df[track_df.frame_id==frame_id]
            img = visualize_one_frame(img, frame_df)

            writer.write(img)

if __name__ == '__main__':
    visualize_neighbors()