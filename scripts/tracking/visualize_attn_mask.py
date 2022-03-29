import os
import os.path as osp
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image

from utils import json_load, json_save, pickle_load, prepare_dir
from data_manager import test_track_map, TEST_TRACK_JSON, RESULT_DIR, DATA_DIR
from object_tracking.library import VideoResult, TrackResult
from object_tracking.utils import (
    ROOT_DIR,
    SAVE_DIR,
    TRAIN_TRACK_DIR,
    TEST_TRACK_DIR,
    get_img_name,
    get_gt_from_idx,
    tracking_config,
)
from object_tracking.utils.visualize import (
    visualize_trajectory,
    apply_mask,
    draw_boxes_on_image,
)

TRAIN_DETECTION_PATH = osp.join(DATA_DIR, "classifier/Centernet2_train_veh_order.json")
TEST_DETECTION_PATH = osp.join(DATA_DIR, "classifier/Centernet2_test_veh_order.json")
RELATION_SAVE_DIR = osp.join(
    RESULT_DIR, "relation_graph", "relation_June22", "final_json_Jun22"
)

input_tracking_dir = prepare_dir(osp.join(RESULT_DIR, "REPORT_VISUALIZE_INPUT"))
vis_tracking_dir = prepare_dir(osp.join(RESULT_DIR, "REPORT_VISUALIZE_TRACKING"))
vis_relation_dir = prepare_dir(osp.join(RESULT_DIR, "REPORT_VISUALIZE_RELATION"))
vis_full_dir = prepare_dir(osp.join(RESULT_DIR, "REPORT_VISUALIZE_FULL"))
vis_denoise_dir = prepare_dir(osp.join(RESULT_DIR, "REPORT_VISUALIZE_DENOISE"))
vis_summarize = prepare_dir(osp.join(RESULT_DIR, "INPUT_SUMMARIZE"))

save_dir = vis_summarize

attn_mask_dir = tracking_config["MASK_SAVE_DIR"]
test_det = json_load(TEST_DETECTION_PATH)

subject_color = "yellow"
follow_color = "green"
follow_by_color = "blue"

good_ids = [75, 77, 79, 84, 156]
best_ids = [94, 98, 121]


def draw_vehicle(img, vehicle: TrackResult, color, is_trajec):
    return visualize_trajectory(img, vehicle.boxes, color, is_trajec)


def get_outside_box(list_boxes, mask):
    tmp = mask.copy()
    tmp[tmp <= 0.3] = 0.0
    choose_boxes = []
    for box in list_boxes:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # if np.sum(tmp[y1:y2+1, x1:x2+1, 0])/((x2-x1)*(y2-y1)) < 0.1:
        if np.sum(tmp[y1 : y2 + 1, x1 : x2 + 1, 0]) < 0.1:
            choose_boxes.append(box)

    return choose_boxes


def main():
    test_data = json_load(TEST_TRACK_JSON)
    list_ids = list(test_data.keys())

    print(f"Save results to {save_dir}")

    for raw_id in tqdm(list_ids[::-1]):
        # for raw_id in tqdm(best_ids):
        new_id = test_track_map[raw_id]
        # if new_id not in best_ids:
        #     continue
        vis_save_path = osp.join(save_dir, f"{new_id}.png")

        relation_path = osp.join(RELATION_SAVE_DIR, f"{new_id}.json")
        vid_data = VideoResult(relation_path)

        # if vid_data.follow is None or vid_data.follow_by is None:
        #     # len(vid_data.follow) == 0 or len(vid_data.follow_by) == 0:
        #     continue

        attn_mask_path = osp.join(attn_mask_dir, f"{new_id}.npy")
        attn_mask = np.load(attn_mask_path)

        # img2vis = vid_data.list_frames[vid_data.n_frames//2]
        idx_frame = 0
        img2vis = vid_data.list_frames[idx_frame]
        img_path = osp.join(DATA_DIR, img2vis)
        pil_img = Image.open(img_path)
        subject = vid_data.get_subject()

        pil_img = draw_vehicle(pil_img, subject, subject_color, True)

        # Apply mask or not
        # pil_img = apply_mask(pil_img, attn_mask)

        # # Visualize follow vehicles
        # for vehicle_id in vid_data.follow:
        #     vehicle = vid_data.get_vehicle(vehicle_id)
        #     pil_img = draw_vehicle(pil_img, vehicle, follow_color, False)

        # # Visualize follow_by vehicles
        # for vehicle_id in vid_data.follow_by:
        #     vehicle = vid_data.get_vehicle(vehicle_id)
        #     pil_img = draw_vehicle(pil_img, vehicle, follow_by_color, False)

        # Visualize outside boxes
        # gt_dict = test_det[str(new_id)]#[idx_frame]
        # frame_info = gt_dict[idx_frame]
        # key = list(frame_info.keys())[0]
        # list_boxes = frame_info[key]
        # out_boxes = get_outside_box(list_boxes, attn_mask)
        # pil_img = draw_boxes_on_image(pil_img, out_boxes, 'red', 5)

        pil_img.save(vis_save_path)
    pass


if __name__ == "__main__":
    main()
