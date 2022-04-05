"""
Script for refining all neighbor tracks based on raw neighbor mapping json generated ny gem_neighbor_mapping.py
Read in the neighbor tracks and the main tracks and refine the relationshop
"""

import json
import numpy as np
from tqdm import tqdm
from utils.bb_utils import refine_boxes, xywh_to_xyxy_lst
from utils.frame_utils import get_frame_ids_by_names
from utils.track_utils import (
    check_is_neighbor_tracks, check_same_tracks, get_relation_between_tracks
)

from utils.constants import (
    AIC22_ROOT, AIC22_META_ROOT,
    TEST_CAM_IDS, TEST_TRACKS_JSON, TEST_AUX_TRACKS_MAPPING_JSON, TEST_AUX_TRACKS_JSON, TEST_AUX_TRACKS_MAPPING_REFINE_JSON,
    TRAIN_CAM_IDS, TRAIN_TRACKS_JSON, TRAIN_AUX_TRACKS_MAPPING_JSON, TRAIN_AUX_TRACKS_JSON, TRAIN_AUX_TRACKS_MAPPING_REFINE_JSON,
    TEST_RELATION_JSON, TRAIN_RELATION_JSON
)

SPLIT = 'test' # or test

if SPLIT == 'train':
    CAM_IDS = TRAIN_CAM_IDS
    TRACKS_JSON = TRAIN_TRACKS_JSON
    OUTPATH = TRAIN_RELATION_JSON
    AUX_TRACKS_MAPPING = TRAIN_AUX_TRACKS_MAPPING_JSON
    AUX_TRACKS = TRAIN_AUX_TRACKS_JSON
else:
    CAM_IDS = TEST_CAM_IDS
    TRACKS_JSON = TEST_TRACKS_JSON
    OUTPATH = TEST_RELATION_JSON
    AUX_TRACKS_MAPPING = TEST_AUX_TRACKS_MAPPING_JSON
    AUX_TRACKS = TEST_AUX_TRACKS_JSON

ANNO = "{AIC22_ROOT}/{FOLDER_NAME}/{CAMERA}/gt/gt.txt"

def run():
    with open(TRACKS_JSON, 'r') as f:
        main_tracks = json.load(f)

    with open(AUX_TRACKS, 'r') as f:
        aux_tracks = json.load(f)

    with open(AUX_TRACKS_MAPPING, 'r') as f:
        aux_tracks_mapping = json.load(f)

    relation_graph = {}

    main_track_ids = list(main_tracks.keys())
    for main_track_id in tqdm(main_track_ids):

        relation_graph[main_track_id] ={
            'follow': [],
            'followed_by': [],
        }

        main_boxes = main_tracks[main_track_id]['boxes']
        main_boxes = xywh_to_xyxy_lst(main_boxes)
        main_frame_names = main_tracks[main_track_id]['frames']
        aux_track_ids = aux_tracks_mapping[main_track_id]

        # Interpolate main track boxes
        main_frame_ids = get_frame_ids_by_names(main_frame_names)
        main_start_id = main_frame_ids[0]
        main_end_id = main_frame_ids[-1]

        main_refined_boxes = refine_boxes(main_frame_ids, main_boxes)
        main_frame_ids = [i for i in range(main_start_id, main_end_id+1)]
        
        if len(main_frame_ids) != len(main_refined_boxes):
            print(len(main_frame_ids))
            print(len(main_refined_boxes))
            asd

        # Interpolate aux track boxes
        for aux_track_id in aux_track_ids:
            aux_boxes = aux_tracks[aux_track_id]['boxes']
            aux_boxes = xywh_to_xyxy_lst(aux_boxes)
            aux_frame_names = aux_tracks[aux_track_id]['frames']
            aux_frame_ids = get_frame_ids_by_names(aux_frame_names)
            aux_frame_ids.sort()

            aux_refined_boxes = refine_boxes(aux_frame_ids, aux_boxes)
            aux_start_id = aux_frame_ids[0]
            aux_end_id = aux_frame_ids[-1]
            aux_frame_ids = [i for i in range(aux_start_id, aux_end_id+1)]

            # Only sampling aux frames and boxes within main track, meaning aux track and main track appear at the same frame
            intersect_frame_ids = list(set(main_frame_ids).intersection(set(aux_frame_ids)))
            intersect_frame_ids.sort()
            main_intersect_boxes = [box for (box, id) in zip(main_refined_boxes, main_frame_ids) if id in intersect_frame_ids]
            aux_intersect_boxes = [box for (box, id) in zip(aux_refined_boxes, aux_frame_ids) if id in intersect_frame_ids]

            # Check if both tracks are the same, both tracks have been aligned
            if check_same_tracks(main_intersect_boxes, aux_intersect_boxes):
                print('iou')
                continue

            # Check if both tracks are related (near to each other)
            if not check_is_neighbor_tracks(main_intersect_boxes, aux_intersect_boxes, dist_mean_threshold=300):
                continue
            
            try:
                # Finally, two determine relation between two neighbor tracks
                relation, avg_distance, avg_cos = get_relation_between_tracks(main_intersect_boxes, aux_intersect_boxes)
            except IndexError as e:
                print('b4 aux_frame_ids ', len(get_frame_ids_by_names(aux_frame_names)))
                print('aux_start_id ', aux_start_id)
                print('aux_end_id ', aux_end_id)
                print('main_frame_ids ', len(np.unique(main_frame_ids)))
                print('intersect_frame_ids ', len(intersect_frame_ids))
                print('aux_frame_ids ', len(np.unique(aux_frame_ids)))
                print('main_refined_boxes, ', len(main_refined_boxes))
                print('aux_refined_boxes ', len(aux_refined_boxes))
                print('main_intersect_boxes ', len(main_intersect_boxes))
                print('aux_intersect_boxes ', len(aux_intersect_boxes))
                asd
            # Store result
            if relation in relation_graph[main_track_id].keys():
                relation_graph[main_track_id][relation].append(aux_track_id)

        if len(relation_graph[main_track_id]['follow']) > 0 or len(relation_graph[main_track_id]['followed_by']) > 0:
            print(relation_graph)
            asd

    with open(OUTPATH, 'w') as f:
        json.dump(relation_graph, f)

if __name__ == '__main__':
    run()