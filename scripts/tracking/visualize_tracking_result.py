"""Visualize tracking result (should run after postprocess_tracking)
"""
import cv2 
import os 
import os.path as osp 
from tqdm import tqdm

from utils import prepare_dir
from object_tracking.library import VideoResult
from object_tracking.utils import (
    tracking_config, class_config, SAVE_DIR
)

track_result_dir = osp.join(SAVE_DIR, tracking_config['EXP_ID'])
save_dir = prepare_dir(osp.join(SAVE_DIR, tracking_config['EXP_ID']+'_VISUALIZE'))

def main():
    for fname in tqdm(os.listdir(track_result_dir)):
        fpath = osp.join(track_result_dir, fname)
        vid_id = fname.split('.')[0]
        if int(vid_id) != 212:
            continue
        save_path = osp.join(save_dir, f'{vid_id}.mp4')
        video_data = VideoResult(fpath)
        video_data.visualize_tracking_result(save_path)
        pass
    pass 

if __name__ == '__main__':
    main() 

