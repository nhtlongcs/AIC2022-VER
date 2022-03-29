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
)
from object_tracking.utils.attn_mask import get_attn_mask
from object_tracking.tools.convert_track import convert_video_track
from object_tracking.library.manager import TrackingManager

TEST_TRACK_ORDERS = list(test_track_map.values())
ATTENTION_THRES = tracking_config["ATTENTION_THRES"]


def concat_feat(vehcol_feats: list, reid_feats: list):
    new_feats = []
    for feat_a, feat_b in zip(vehcol_feats, reid_feats):
        new_feats.append(np.concatenate([feat_a, feat_b], axis=0))

    return new_feats


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


def tracking(
    config: dict,
    ds_config: dict,
    json_save_dir: str,
    mask_dir: str = None,
    vis_save_dir: str = None,
    verbose=False,
):
    mode_json_dir = json_save_dir
    gt_dict = json_load(config["track_dir"])
    list_keys = list(gt_dict.keys())

    for track_order in tqdm(list_keys):
        track_order = str(track_order)
        save_json_path = osp.join(json_save_dir, f"{track_order}.json")

        img_dict = gt_dict[track_order]
        img_names = get_img_name(img_dict)
        feat_path = osp.join(config["feat_dir"][0], f"{track_order}.pkl")
        reid_feat = pickle_load(feat_path)
        vehcol_path = osp.join(config["feat_dir"][1], f"{track_order}.pkl")
        vehcol_feat = pickle_load(vehcol_path)

        attn_mask = None
        if mask_dir is not None:
            attn_mask = np.load(osp.join(mask_dir, f"{track_order}.npy"))

        ans = {}
        # Initialize deep sort.
        deepsort = TrackingManager(ds_config)

        if config["save_video"]:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            save_visualize_path = os.path.join(vis_save_dir, f"{track_order}.avi")
            out = None

        frame = None
        for i in range(len(img_names)):
            img_path = osp.join(DATA_DIR, img_names[i])
            frame = cv2.imread(img_path)
            frame = frame.astype(np.uint8)

            if i == 0 and config["save_video"]:
                h, w, c = frame.shape
                out = cv2.VideoWriter(save_visualize_path, fourcc, 1, (w, h))

            detections, out_scores = get_gt_from_idx(
                i, gt_dict[track_order]
            )  # detections: list of xywh boxes
            detections = np.array(detections)
            out_scores = np.array(out_scores)

            vehcol_features = vehcol_feat[img_names[i]]
            reid_features = reid_feat[img_names[i]]
            new_feats = concat_feat(vehcol_features, reid_features)
            features = new_feats

            # Load attention mask to remove unrelated objects
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

            # print('='*20 + f' Tracking step {i} ' + '='*20)
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

                track_dict["box"] = [
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                ]
                track_dict["feature"] = feature
                track_list.append(track_dict)

                if config["save_video"] and (frame is not None):
                    # Draw bbox from tracker.
                    cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        str(id_num),
                        (int(bbox[0]), int(bbox[1])),
                        0,
                        5e-3 * 200,
                        (0, 255, 0),
                        2,
                    )

            ans[img_names[i]] = track_list
            if config["save_video"]:
                out.write(frame)

        if config["save_video"]:
            out.release()

        save_json_path = os.path.join(mode_json_dir, f"{track_order}.json")
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

    for mode in config:
        mask_save_dir = prepare_dir(tracking_config["MASK_SAVE_DIR"])
        vis_save_dir = prepare_dir(tracking_config["VIS_SAVE_DIR"])
        tracking_save_dir = prepare_dir(tracking_config["TRACK_SAVE_DIR"])

        # 1. Get Attention map
        print(f">> Extract attention map for tracking, save results to {mask_save_dir}")
        test_data = json_load(TEST_TRACK_JSON)
        list_ids = list(test_data.keys())
        for raw_id in tqdm(list_ids):
            new_id = test_track_map[raw_id]
            save_path = osp.join(mask_save_dir, f"{new_id}.npy")

            if osp.isfile(save_path):
                vid_attn_mask = np.load(save_path)
            else:
                vid_data = test_data[raw_id]
                vid_attn_mask, is_interpolate = get_attn_mask(
                    vid_data,
                    tracking_config["ATN_MASK"]["EXPAND_RATIO"],
                    tracking_config["ATN_MASK"]["N_EXPAND"],
                )
                np.save(save_path, vid_attn_mask)

        # 2. Run Tracking using attention masks
        print(f">> Run DeepSort on {mode} mode, save result to {tracking_save_dir}")
        tracking(
            config[mode],
            tracking_config,
            tracking_save_dir,
            mask_save_dir,
            vis_save_dir,
            verbose=False,
        )

