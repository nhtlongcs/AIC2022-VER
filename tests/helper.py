import os
import cv2
import numpy as np
import random
import uuid
from glob import glob
import json


def create_dummy_image(im_path, IMG_SIZE=512):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    img[0:IMG_SIZE, 0:IMG_SIZE] = (255, 255, 255)
    cv2.imwrite(im_path, img)


def dummy_box(IMG_SIZE=512):
    return [
        random.randint(0, IMG_SIZE),
        random.randint(0, IMG_SIZE),
        random.randint(0, IMG_SIZE),
        random.randint(0, IMG_SIZE),
    ]


dummy_tracklet = {
    "S01": ["c001", "c002"],
    "S02": ["c003", "c004"],
    "S03": ["c005", "c006"],
    "S04": ["c007", "c008"],
}


def gen_bk(bk_folder):
    for s, c_ls in dummy_tracklet.items():
        for c in c_ls:
            img_path = os.path.join(bk_folder, f"{s}_{c}.jpg")
            create_dummy_image(img_path)


def gen_dummy_frames(frames_folder):
    for data_dir, s_ls in {
        "train": ["S01", "S02"],
        "validation": ["S03", "S04"],
    }.items():
        for s in s_ls:
            c_ls = dummy_tracklet[s]
            for c in c_ls:
                img_dir = os.path.join(frames_folder, data_dir, s, c, "img1")
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, "000001.jpg")
                create_dummy_image(img_path)


def gen_dummy_tracklet(img_ls):
    track_imgs = img_ls[: random.randint(1, len(img_ls))]
    track_bboxes = [dummy_box() for _ in range(len(track_imgs))]
    uuid_str = str(uuid.uuid4())

    return (
        uuid_str,
        {
            "frames": track_imgs,
            "boxes": track_bboxes,
            "nl": ["abc", "def", "ghi"],
            "nl_other_views": ["def", "ghi", "abc"],
        },
    )


def gen_dummy_tracklet_json(root_dir):
    # gen dummy tracklet json train and val
    img_ls = glob(os.path.join(root_dir, "extracted_frames", "*/*/*/img1/*.jpg"))
    img_ls = [
        "./" + img_path[img_path.find("extracted_frames") + len("extracted_frames/") :]
        for img_path in img_ls
    ]
    uuid_ls = []
    for data_dir, filename in [("train", "train"), ("validation", "val")]:
        tracklet_path = os.path.join(root_dir, f"{filename}.json")
        with open(tracklet_path, "w") as f:
            tmp_tracklets = {}
            for _ in range(10):
                uuid, tracklet = gen_dummy_tracklet(img_ls)
                tmp_tracklets[uuid] = tracklet
                uuid_ls.append(uuid)
            f.write(json.dumps(tmp_tracklets))
    return uuid_ls


def gen_motion(motion_map_folder, uuid_ls):
    for uuid_str in uuid_ls:
        img_path = os.path.join(motion_map_folder, uuid_str + ".jpg")
        create_dummy_image(img_path)


def create_fake_data():
    os.makedirs("data/meta/", exist_ok=True)

    os.makedirs("data/meta/bk_map", exist_ok=True)
    gen_bk("data/meta/bk_map")

    os.makedirs("data/meta/extracted_frames/", exist_ok=True)
    gen_dummy_frames("data/meta/extracted_frames")

    uuid_ls = gen_dummy_tracklet_json("data/meta")

    os.makedirs("data/meta/motion_map", exist_ok=True)
    gen_motion("data/meta/motion_map", uuid_ls)

