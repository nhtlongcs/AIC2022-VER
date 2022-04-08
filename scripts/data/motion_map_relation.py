from dataclasses import asdict
import glob
import json
import multiprocessing
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp

n_worker = multiprocessing.cpu_count() // 2
meta_data_path = sys.argv[1]
root = Path(meta_data_path) / 'extracted_frames'
save_bk_dir = Path(meta_data_path) / 'bk_map'
save_rel_mo_dir = Path(meta_data_path) / 'rel_motion_map'

test_track_path = osp.join(meta_data_path,"test_tracks.json")
train_track_path = osp.join(meta_data_path,"train_tracks.json")
train_relation_track_path = osp.join(meta_data_path, "relation", "train_relation.json")
test_relation_track_path = osp.join(meta_data_path, "relation", "test_relation.json")

test_neighbor_track_path = osp.join(meta_data_path, "relation", "test_neighbors.json")
train_neighbor_track_path = osp.join(meta_data_path, "relation", "train_neighbors.json")

assert test_track_path and train_track_path, "Paths to test and train tracks are not provided"

with open(test_track_path) as f:
    tracks_test = json.load(f)
with open(train_track_path) as f:
    tracks_train = json.load(f)

all_tracks = tracks_test
for track in tracks_train.keys():
    all_tracks[track] = tracks_train[track]


# Append neighbors to all tracks
## All neighbors mapping

with open(train_relation_track_path) as f:
    rel_train = json.load(f)
with open(test_relation_track_path) as f:
    rel_test = json.load(f)

all_rel_mapping = rel_test
all_rel_mapping.update(rel_train)

## Read in all neighbor tracks info
with open(train_neighbor_track_path) as f:
    rel_tracks_train = json.load(f)
with open(test_neighbor_track_path) as f:
    rel_tracks_test = json.load(f)

all_rel_tracks = rel_tracks_test
all_rel_tracks.update(rel_tracks_train)

save_rel_mo_dir.mkdir(exist_ok=True)

def get_bk_map(info):
    # weighted bk map?
    path, save_name = info
    img = glob.glob(path + "/img1/*.jpg")
    img.sort()
    interval = min(5, max(1, int(len(img) / 200)))
    img = img[::interval][:10]
    # img = img[::interval][:1000]
    imgs = []
    outpath = save_bk_dir / f"{save_name}.jpg"
    if outpath.exists():
        return
    for name in img:
        imgs.append(cv2.imread(name))
    avg_img = np.mean(np.stack(imgs), 0)
    avg_img = avg_img.astype(np.int64)
    cv2.imwrite(str(outpath), avg_img)


def get_rel_motion_map(info):

    ALPHA = 0.3 # for blending
    FOLLOWEE_RATE = 0.33 # reduce to get longer tracklet of followeee
    FOLLOWER_RATE = 3 # increase to get longer tracklet of follower
    
    track_id = info['track_id']
    track_info = info['track_info']
    followee_tracks = info['followees'] 
    follower_tracks = info['followers']
    num_frames = len(track_info["frames"])
    sampling_rate = max(len(track_info["frames"]) // 15, 2)

    # BK map
    avg_filename = str(
        Path(track_info["frames"][0]).parent.parent.parent.name  # S01
        + "_"
        + Path(track_info["frames"][0]).parent.parent.name  # c001
        + ".jpg"
    )
    avg_img = cv2.imread(str(save_bk_dir / avg_filename)).astype(np.int64)

    # Main track motion
    for i in range(num_frames):
        frame_path = track_info["frames"][i]
        frame_path = root / frame_path
        assert frame_path.exists(), "Frame path does not exist"
        frame = cv2.imread(str(frame_path))
        box = track_info["boxes"][i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        x2, y2 = x+w, y+h
        if i == 0:
            example = np.zeros(frame.shape, np.int64)
        if i % sampling_rate == 0:
            example[y:y2,x:x2,:] = ALPHA*example[y:y2,x:x2,:] + (1-ALPHA) * frame[y:y2,x:x2,:]

    ## Paste background into tracklet motion
    postions = (
        (example[:, :, 0] == 0) & (example[:, :, 1] == 0) & (example[:, :, 2] == 0)
    )
    example[postions] = avg_img[postions]
    cv2.imwrite(str(save_rel_mo_dir / 'main' / f"{track_id}.jpg"), example)

    # Followee motion map
    followee_example = np.zeros(frame.shape, np.int64)
    for followee_track in followee_tracks:
        _, intersected_followee_ids, intersected_main_ids = np.intersect1d(
            followee_track['frames'],
            track_info['frames'],
            return_indices=True
        )


        middle_frame_pos = int(len(intersected_followee_ids) / FOLLOWEE_RATE)
  
        for i in range(len(intersected_followee_ids)):

            if i % sampling_rate == 0:
                if i >= middle_frame_pos:
                    box = track_info["boxes"][intersected_main_ids[i]]
                    frame_path = track_info["frames"][intersected_main_ids[i]]
                    frame_path = root / frame_path
                    assert frame_path.exists(), "Frame path does not exist"
                    frame = cv2.imread(str(frame_path))
                else:
                    box = followee_track["boxes"][intersected_followee_ids[i]]
                    frame_path = followee_track["frames"][intersected_followee_ids[i]]
                    frame_path = root / frame_path
                    assert frame_path.exists(), "Frame path does not exist"
                    frame = cv2.imread(str(frame_path))

                x,y,w,h = box[0], box[1], box[2], box[3]
                x2, y2 = x+w, y+h
                
                followee_example[y:y2,x:x2,:] = (
                    ALPHA*followee_example[y:y2,x:x2,:] + 
                    frame[y:y2,x:x2,:] * (1-ALPHA)
                )

    ## Paste background into tracklet motion
    postions = (
        (followee_example[:, :, 0] == 0) & (followee_example[:, :, 1] == 0) & (followee_example[:, :, 2] == 0)
    )
    followee_example[postions] = avg_img[postions]
    cv2.imwrite(str(save_rel_mo_dir / 'followees' / f"{track_id}.jpg"), followee_example)

    # Follower motion map
    follower_example = np.zeros(frame.shape, np.int64)
    for follower_track in follower_tracks:
        _, intersected_follower_ids, intersected_main_ids = np.intersect1d(
            follower_track['frames'],
            track_info['frames'],
            return_indices=True
        )

        middle_frame_pos = int(len(intersected_follower_ids) / FOLLOWER_RATE)

        for i in range(len(intersected_follower_ids)):

            if i % sampling_rate == 0:
                if i < middle_frame_pos:
                    box = track_info["boxes"][intersected_main_ids[i]]
                    frame_path = track_info["frames"][intersected_main_ids[i]]
                    frame_path = root / frame_path
                    assert frame_path.exists(), "Frame path does not exist"
                    frame = cv2.imread(str(frame_path))
                else:
                    box = follower_track["boxes"][intersected_follower_ids[i]]
                    frame_path = follower_track["frames"][intersected_follower_ids[i]]
                    frame_path = root / frame_path
                    assert frame_path.exists(), "Frame path does not exist"
                    frame = cv2.imread(str(frame_path))

                x,y,w,h = box[0], box[1], box[2], box[3]
                x2, y2 = x+w, y+h
                
                follower_example[y:y2,x:x2,:] = (
                    ALPHA*follower_example[y:y2,x:x2,:] + 
                    frame[y:y2,x:x2,:] * (1-ALPHA)
                )

    ## Paste background into tracklet motion
    postions = (
        (follower_example[:, :, 0] == 0) & (follower_example[:, :, 1] == 0) & (follower_example[:, :, 2] == 0)
    )
    follower_example[postions] = avg_img[postions]
    cv2.imwrite(str(save_rel_mo_dir / 'followers' / f"{track_id}.jpg"), follower_example)

def parallel_task(task, files):
    with multiprocessing.Pool(n_worker) as pool:
        for imgs in tqdm(pool.imap_unordered(task, files)):
            pass


def extract_bk_map():
    paths = root.glob("*/*/*")
    files = []
    for path in paths:
        files.append((str(path), path.parent.name + "_" + path.name))

    parallel_task(get_bk_map, files)


def extract_mo_map():
    files = []
    for track_id in all_tracks.keys():
        neighbor_lst = all_rel_mapping[track_id]

        follower_tracks = [all_rel_tracks[neighbor_id] for neighbor_id in neighbor_lst['follow']] 
        followee_tracks = [all_rel_tracks[neighbor_id] for neighbor_id in neighbor_lst['followed_by']] 

        files.append({
            'track_id': track_id,
            'track_info': all_tracks[track_id],
            'followees': followee_tracks,
            'followers': follower_tracks,
        })
    parallel_task(get_rel_motion_map, files)

    # for file in tqdm(files):
        # get_rel_motion_map(file)


if __name__ == "__main__":
    # extract_bk_map()
    extract_mo_map()
