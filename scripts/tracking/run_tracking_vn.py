import os
import os.path as osp
import cv2
from tqdm import tqdm
import numpy as np

from utils import json_load, json_save, pickle_load, prepare_dir
from data_manager import test_track_map, TEST_TRACK_JSON, RESULT_DIR, DATA_DIR
from object_tracking.utils import (
    ROOT_DIR,
    SAVE_DIR,
    TRAIN_TRACK_DIR,
    TEST_TRACK_DIR,
    get_img_name,
    get_gt_from_idx,
    tracking_config,
    class_config,
)
from object_tracking.utils.attn_mask import get_attn_mask
from object_tracking.tools.convert_track import convert_video_track
from object_tracking.library.manager import TrackingManager
from object_tracking.library import VideoResult

from classifier import ClassifierManager

TEST_TRACK_ORDERS = list(test_track_map.values())
ATTENTION_THRES = tracking_config["ATTENTION_THRES"]
DATA_DIR = "/content/video_3"


def get_detection_to_track(attn_mask: np.ndarray, detections: np.array):
    chosen_ids = []
    for i, box in enumerate(detections):
        x, y, w, h = box
        area = w * h
        mask_area = attn_mask[y : y + h + 1, x : x + w + 1, 0]
        mask_area[np.where(mask_area <= ATTENTION_THRES)] = 0
        overlap_ratio = np.sum(mask_area) / area

        if overlap_ratio > ATTENTION_THRES:
            chosen_ids.append(i)

    return chosen_ids


def xyxy_to_xywh(box: list):
    res = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
    return res


def tracking(
    config: dict,
    vid_img_dir: str,
    feat_tracking: dict,
    det_data: dict,
    ds_config: dict,
    save_json_path: str,
    mask_dir: str = None,
):

    list_fnames = list(feat_tracking.keys())
    print(f"list_fnames: {list_fnames}")

    attn_mask = None
    if mask_dir is not None:
        attn_mask = np.load(osp.join(mask_dir, f"{track_order}.npy"))

    ans = {}

    # Initialize deep sort.
    deepsort = TrackingManager(ds_config)

    frame = None
    for i in tqdm(range(len(list_fnames))):
        img_name = list_fnames[i]
        img_path = osp.join(vid_img_dir, img_name)

        frame = cv2.imread(img_path)
        frame = frame.astype(np.uint8)

        detections = det_data[img_name]
        detections = [xyxy_to_xywh(det) for det in detections]
        out_scores = [1.0] * len(detections)
        # detections, out_scores = get_gt_from_idx(i, gt_dict[track_order])# detections: list of xywh boxes
        detections = np.array(detections)
        out_scores = np.array(out_scores)

        features = feat_tracking[img_name]  # 2048

        if attn_mask is not None:
            chosen_ids = get_detection_to_track(attn_mask, detections)
            dets, scores, feats = [], [], []

            for idx in chosen_ids:
                dets.append(detections[idx])
                scores.append(out_scores[idx])
                feats.append(features[idx])

            detections = np.array(dets)
            out_scores = np.array(scores)
            features = np.array(feats)

        tracker, detections_class = deepsort.run_deep_sort(
            out_scores, detections, features
        )

        track_list = []
        count = 0

        for track in tracker.tracks:
            count += 1
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            id_num = str(track.track_id)  # Get the ID for the particular track.
            feature = (
                track.last_feature
            )  # Get the feature vector corresponding to the detection.
            track_dict = {}
            track_dict["id"] = id_num

            track_dict["box"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            track_dict["feature"] = feature
            track_list.append(track_dict)

        ans[img_name] = track_list

    reformat_res = convert_video_track(
        ans, save_json_path, save_feat=config["save_feat"]
    )

    pass


if __name__ == "__main__":
    config = {
        # "train_total": {
        #     "track_dir": TRAIN_TRACK_DIR,
        #     "mode": "train_total",
        #     "save_video": False,
        #     "save_feat": False,
        # },
        "test": {
            "track_dir": TEST_TRACK_DIR,
            "feat_dir": [
                osp.join(SAVE_DIR, "reid/test_feat_resnet101_ibn_a"),
                osp.join(RESULT_DIR, "classifier/test_feat_tracking"),
            ],
            "save_video": False,
            "save_feat": False,
        }
    }

    # video_names = ['video_4', 'video_5', 'video_6']
    video_names = ["video_7"]
    feat_tracking_dir = "/content/AI_City_2021/results/vn_example/feat_tracking"
    det_dir = "/content/AI_City_2021/results/vn_example/det_results"
    save_dir = "/content/AI_City_2021/results/vn_example/result_tracking"
    img_dir = "/content/AI_City_2021/dataset/vn_example/results"

    for video_id in video_names:
        print(f">> Run {video_id}")
        feat_tracking_path = osp.join(feat_tracking_dir, f"{video_id}.pkl")
        det_json = osp.join(det_dir, f"{video_id}.json")
        save_json_path = osp.join(save_dir, f"{video_id}.json")
        vid_img_dir = osp.join(img_dir, video_id)

        feat_tracking = pickle_load(feat_tracking_path)
        det_data = json_load(det_json)
        mode = "test"

        det_data_2 = {}
        for sample in det_data:
            det_data_2.update(sample)

        # 1. Run tracking
        tracking(
            config[mode],
            vid_img_dir,
            feat_tracking,
            det_data_2,
            tracking_config,
            save_json_path,
            mask_dir=None,
        )
        # if not osp.isfile(save_json_path):
        #     tracking(config[mode], vid_img_dir, feat_tracking, det_data_2, tracking_config, save_json_path, mask_dir=None)

        # 2. Show tracking result
        # vis_path = f'/content/AI_City_2021/results/vn_example/result_tracking/{video_id}_subject.mp4'
        video_data = VideoResult(save_json_path, vid_img_dir)
        video_data.visualize(
            save_path=f"/content/AI_City_2021/results/vn_example/result_tracking/{video_id}_tracking.mp4"
        )
        # video_data.visualize_tracking_result(
        #     save_path=f'/content/AI_City_2021/results/vn_example/result_tracking/{video_id}_tracking.mp4',
        #     is_draw_subject=False
        # )

        print(">> Finish")
        pass

    # video_id = 'video_30'

    # tracking_save_dir = f''
    # mask_save_dir = None

    # feat_tracking_path = f'/content/AI_City_2021/results/vn_example/feat_tracking/{video_id}.pkl'
    # det_json = f'/content/AI_City_2021/results/vn_example/{video_id}.json'

    # save_json_path = f'/content/AI_City_2021/results/vn_example/result_tracking/{video_id}.json'

    # feat_tracking = pickle_load(feat_tracking_path)
    # det_data = json_load(det_json)
    # mode='test'

    # det_data_2 = {}
    # for sample in det_data:
    #     det_data_2.update(sample)

    # # 1. Run tracking
    # tracking(config[mode], feat_tracking, det_data_2, tracking_config, save_json_path)

    # # 2. Show tracking result
    # vis_path = f'/content/AI_City_2021/results/vn_example/result_tracking/{video_id}_subject.mp4'
    # data_dir = DATA_DIR
    # video_data = VideoResult(save_json_path, data_dir)
    # video_data.visualize(vis_path)

