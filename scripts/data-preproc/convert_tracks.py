import argparse
import cv2
import json
import os
import os.path as osp
from tqdm import tqdm

JSON_FILE = "K:/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/train_tracks.json"
IMAGE_DIR = "K:/Github/AIC2022-VER/data/meta/extracted_frames"
OUT_DIR = 'K:/Github/AIC2022-VER/visualize_tool/track_vids/train'

def draw_bbox(image, boxes, labels=None) -> None:
        tl = int(round(0.001 * max(image.shape[:2])))  # line thickness
        
        tup = zip(boxes, labels) if labels is not None else boxes

        for item in tup:
            if labels is not None:
                box, label = item
            else:
                box, label = item, None
            color = [0, 255, 0]

            coord = [box[0], box[1], box[2], box[3]]
            c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
            cv2.rectangle(image, c1, c2, color, thickness=tl*2)

            if label is not None:
                tf = max(tl - 2, 1)  # font thickness
                s_size = cv2.getTextSize(f'{label}', 0, fontScale=float(tl) / 3, thickness=tf)[0]
                c2 = c1[0] + s_size[0] + 15, c1[1] - s_size[1] - 3
                cv2.rectangle(image, c1, c2, color, -1)  # filled
                cv2.putText(image, f'{label}', (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                            thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

def extract_tracks(image_dir, json_file, out_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    track_ids = sorted(list(data.keys()))
    order_dict = {}
    for idx, track_id in enumerate(tqdm(track_ids)):
        order_dict[track_id] = idx


        track_info = data[track_id]
        frame_paths = track_info['frames']
        boxes = track_info['boxes']


        image = cv2.imread(osp.join(image_dir, frame_paths[0]))
        h, w, _ = image.shape
        
        # Init video writer
        FPS = 10
        outpath = osp.join(out_dir, str(idx) + '.avi')
        outvid = cv2.VideoWriter(
            outpath,   
            cv2.VideoWriter_fourcc("M", "J", "P", "G"), 
            FPS, 
            (w, h))


        for frame_path, box in zip(frame_paths, boxes):
            frame_path = osp.join(image_dir, frame_path)
            image = cv2.imread(frame_path)

            box[2] += box[0]
            box[3] += box[1]
            draw_bbox(image, [box], labels=[str(idx)])
            outvid.write(image)

        break
    with open(osp.join(out_dir, 'order.json'), 'w') as f:
        json.dump(order_dict, f, indent=4)

if __name__ == '__main__':
    extract_tracks(IMAGE_DIR, JSON_FILE, OUT_DIR)