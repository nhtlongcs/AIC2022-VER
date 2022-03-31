import os
import os.path as osp
import json
from turn_detector import TurnDetector
from tqdm import tqdm


if __name__ == '__main__':

    detector = TurnDetector(
        eps = 0.035,
        skip_frame = 5,
        is_normalize=0 # increase means more retrieved
    )

    with open("/home/kaylode/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/test_tracks.json", 'r') as f:
        data = json.load(f)

    turn_ids = []
    # fail_ids = []
    new_not_old = []
    norm_area_dict = {}
    
    vertical_views = {}
    list_straight_views = []
    straight_percent_dict = {}
    speed_dict = {}

    for track_id, track_value in tqdm(data.items()):
        boxes = track_value['boxes']

        is_turn, turn_state, norm_area, list_points = detector.process(boxes)
        is_vertical_view, straight_percent, list_angles, speed_record = detector.find_vertical_views(list_points)

        vertical_views[track_id] = list_angles
        straight_percent_dict[track_id] = straight_percent
        speed_dict[track_id] = speed_record
        norm_area_dict[track_id] = norm_area
        
        if is_turn:
            turn_ids.append(track_id)
            if is_vertical_view:
                list_straight_views.append(track_id)
    
    print(turn_ids)
    print(len(turn_ids))
    # print(len(list_straight_views))