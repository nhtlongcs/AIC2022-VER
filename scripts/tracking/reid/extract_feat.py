import json, pickle
import os
import os.path as osp
import cv2
from tqdm import tqdm

from external.object_tracking.torchreid.utils import FeatureExtractor

from config.tracking import MODEL_NAME, WEIGHT_DIR

extractor = FeatureExtractor(
    model_name=MODEL_NAME,
    model_path=WEIGHT_DIR,  # osp.join(WEIGHT_DIR, MODEL_NAME, 'model.pth.tar-25'),
    device="cpu",
)

SAVE_DIR = "./results"
TRAIN_TRACK_JSON = (
    "/content/AI_City_2021/classifier/data/Centernet2_train_veh_order.json"
)
TEST_TRACK_JSON = "/content/AI_City_2021/classifier/data/Centernet2_test_veh_order.json"
DATA_DIR = "/content/AI_City_2021/dataset"
SAVE_PERIOD = 10

os.makedirs(SAVE_DIR, exist_ok=True)
train_track = json.load(open(TRAIN_TRACK_JSON))
test_track = json.load(open(TEST_TRACK_JSON))
data_track = {"train": train_track, "test": test_track}


def pickle_save(data, save_path, verbose=False):
    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    if verbose:
        print(f"save result to {save_path}")


def extract_feature(data_track, data_dir, mode_save_dir: str):
    feat = {}
    count = 1
    list_keys = list(data_track.keys())
    print(f"Extract {len(list_keys)} tracks")
    for key_track in tqdm(list_keys):
        count += 1
        track_save_path = osp.join(mode_save_dir, f"{key_track}.pkl")
        if osp.isfile(track_save_path):
            continue

        track_feat = {}
        for frame_dict in data_track[key_track]:
            frame_path = list(frame_dict.keys())[0]
            list_boxes = []

            for box_coor in frame_dict[frame_path]:
                img_path = osp.join(data_dir, frame_path)
                cv_img = cv2.imread(img_path)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

                box_coor = [int(x) for x in box_coor]
                x_0, y_0, x_1, y_1 = box_coor
                crop = cv_img[y_0:y_1, x_0:x_1, :]
                list_boxes.append(crop)

            track_feat[frame_path] = extractor(list_boxes).detach().cpu().numpy()

        # print(f'Extract {count}th')
        pickle_save(track_feat, track_save_path)
        feat[key_track] = track_feat

    return feat


def test():
    feat_path = "/home/ntphat/projects/AI_City_2021/object_tracking/reid/results/train_feat_tracking/0.pkl"
    feat = None
    with open(feat_path, "rb") as f:
        feat = pickle.load(f)

    for k in feat:
        print(f"{k}: {feat[k].shape}")


if __name__ == "__main__":
    for mode in ["train", "test"]:
        # for mode in ["test"]:
        print(f"Extract in {mode} data")
        save_path = osp.join(SAVE_DIR, f"{mode}_feat.pkl")
        mode_save_dir = osp.join(SAVE_DIR, f"{mode}_feat_resnet101_ibn_a")
        print(f"Save result to {mode_save_dir}")
        os.makedirs(mode_save_dir, exist_ok=True)

        feat = extract_feature(data_track[mode], DATA_DIR, mode_save_dir)
        pickle_save(feat, save_path)

