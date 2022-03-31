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
    # Init classifier for veh/col prediction
    class_manager = ClassifierManager()
    print(f"Init classifier successfully")

    # video_names = ['video_4', 'video_5', 'video_6']
    video_names = ["video_7"]
    feat_tracking_dir = "/content/AI_City_2021/results/vn_example/feat_tracking"
    det_dir = "/content/AI_City_2021/results/vn_example/det_results"
    tracking_dir = "/content/AI_City_2021/results/vn_example/result_tracking"
    save_dir = "/content/AI_City_2021/results/vn_example/result_post_tracking"
    img_dir = "/content/AI_City_2021/dataset/vn_example/results"

    for video_id in video_names:
        print(f">> Run {video_id}")
        det_json = osp.join(det_dir, f"{video_id}.json")
        save_path = osp.join(save_dir, f"{video_id}.json")
        vid_img_dir = osp.join(img_dir, video_id)
        vid_tracking_path = osp.join(tracking_dir, f"{video_id}.json")

        video_data = VideoResult(vid_tracking_path, vid_img_dir)

        # 2. Find stop vehicles
        stop_tracks = find_stop_track(video_data)
        video_data.set_stop_tracks(stop_tracks)

        # 3. set class name
        video_data.set_class_names(
            class_manager,
            veh_thres=class_config["VEH_THRES"],
            col_thres=class_config["COL_THRES"],
        )
        video_data.to_json(save_path)

        # 4. visualize result
        vis_save_path = osp.join(save_dir, f"{video_id}.mp4")
        video_data.visualize(vis_save_path)
        pass

    # video_id = 'video_30'
    # data_dir = '/content/video_3'
    # save_dir = '/content/AI_City_2021/results/vn_example/result_post_tracking'
    # video_json = f'/content/AI_City_2021/results/vn_example/result_tracking/{video_id}.json'
    # save_path = osp.join(save_dir, f'{video_id}.json') #f'/content/AI_City_2021/results/vn_example/result_post_tracking/{video_id}.json'

    # video_data = VideoResult(video_json, data_dir)

    # # 2. Find stop vehicles
    # stop_tracks = find_stop_track(video_data)
    # video_data.set_stop_tracks(stop_tracks)

    # # 3. set class name
    # video_data.set_class_names(class_manager, veh_thres=class_config['VEH_THRES'], col_thres=class_config['COL_THRES'])
    # video_data.to_json(save_path)

    # # 4. visualize result
    # vis_save_path = osp.join(save_dir, f'{video_id}.mp4')
    # video_data.visualize(vis_save_path)
    pass


if __name__ == "__main__":
    main()
