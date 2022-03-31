import os
import os.path as osp
import json
from stop_detector import StopDetector
from tqdm import tqdm


if __name__ == '__main__':

    detector = StopDetector(
        k = 5,
        delta = 10, # as paper
        alpha=0.15 # increase means more retrieved
    )

    with open("/home/kaylode/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/test_tracks.json", 'r') as f:
        data = json.load(f)

    stop_ids = []
    fail_ids = []

    for track_id, track_value in tqdm(data.items()):
        boxes = track_value['boxes']

        is_stop = detector.process(boxes)
        if is_stop:
            stop_ids.append(track_id)
    
    print(stop_ids)
    print(len(stop_ids))