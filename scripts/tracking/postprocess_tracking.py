import os
import os.path as osp
from tqdm import tqdm
import sys

from utils import create_logger
from utils import json_load, prepare_dir
from data_manager import test_track_map, TEST_TRACK_JSON, RESULT_DIR, DATA_DIR
from object_tracking.library import VideoResult

from object_tracking.utils import (
    find_stop_track,
    xywh_to_xyxy,
    find_subject_track,
    SAVE_DIR,
    subject_config,
    stop_config,
    tracking_config,
    class_config,
)
from classifier import ClassifierManager

VISUALIZE = False
EXP_ID = tracking_config["EXP_ID"]


def main():
    post_process_save_dir = prepare_dir(osp.join(SAVE_DIR, EXP_ID))
    vis_post_process_save_dir = prepare_dir(osp.join(SAVE_DIR, f"vis_{EXP_ID}"))

    print(f"Save result to {post_process_save_dir}")
    print(f"Save visualize result to {vis_post_process_save_dir}")

    track_save_dir = tracking_config["TRACK_SAVE_DIR"]
    test_data = json_load(TEST_TRACK_JSON)

    # Init classifier for veh/col prediction
    class_manager = ClassifierManager()
    print(f"Init classifier successfully")
    no_subject = []

    for fname in tqdm(os.listdir(track_save_dir)):
        fpath = osp.join(track_save_dir, fname)
        new_id = fname.split(".")[0]
        save_path = osp.join(post_process_save_dir, f"{new_id}.json")
        vis_save_path = osp.join(vis_post_process_save_dir, f"{new_id}.avi")

        old_id = test_track_map[new_id]

        # print(f'Run track {new_id} - {old_id}')
        gt_boxes = test_data[old_id]["boxes"]
        gt_boxes = [xywh_to_xyxy(box) for box in gt_boxes]
        vid_data = VideoResult(fpath)
        vid_data.gt_boxes = gt_boxes

        # 1. Find subject
        # print('='*15 + f' 1. Find subject ' + '='*15)
        subject_track_id, score_dict = find_subject_track(vid_data, gt_boxes)
        if subject_track_id is not None:
            vid_data.set_subject(subject_track_id)
            # print(score_dict)
        else:
            # Remove fail track ids
            remove_track_ids = [rec["track_id"] for rec in score_dict]
            for track_id in remove_track_ids:
                vid_data.remove_track(track_id)
            no_subject.append(new_id)

            subject_track_id = remove_track_ids[0]
            vid_data.set_default_subject(gt_boxes, subject_track_id)
            print(
                f"[Video {new_id}] Remove fail track ids: {remove_track_ids}, use {subject_track_id} to create default subject from groundtruth boxes"
            )
            pass

        # 2. Find stop vehicles
        stop_tracks = find_stop_track(vid_data)
        vid_data.set_stop_tracks(stop_tracks)

        # 3. set class name
        vid_data.set_class_names(
            class_manager,
            veh_thres=class_config["VEH_THRES"],
            col_thres=class_config["COL_THRES"],
        )
        vid_data.to_json(save_path)
        if VISUALIZE:
            vid_data.visualize(vis_save_path)
        pass

    print(f"No subject video, {len(no_subject)} ={no_subject}")
    pass


if __name__ == "__main__":
    main()
    pass
