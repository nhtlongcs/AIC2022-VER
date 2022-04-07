import numpy as np

def bb_intersection_over_union(boxA, boxB):
    """Calculate IOU
    box in (x1, y1, x2, y2)"""
    
    assert boxA[2] >= boxA[0] and boxA[3] >= boxA[1], "wrong format"
    assert boxB[2] >= boxB[0] and boxB[3] >= boxB[1], "wrong format"

	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def interpolate_box(cur_idx, start_idx, end_idx, start_box, end_box):
    """
    Interpolate box at current frame index
    """
    end_ratio = (end_idx-cur_idx)/(end_idx-start_idx)
    start_ratio = (cur_idx-start_idx)/(end_idx-start_idx)
    
    box1, box2 = start_box, end_box
    if isinstance(start_box, list):
        box1 = np.array(start_box) 
    if isinstance(end_box, list):
        box2 = np.array(end_box) 
    
    cur_box = start_ratio*box2 + end_ratio*box1 
    cur_box = cur_box.tolist()
    return cur_box

def generate_boxes(start_idx, end_idx, start_box, end_box):
    """
    Interpolate boxes at between timestamp
    """
    res = [] 
    for i in range(start_idx+1, end_idx):
        res.append(interpolate_box(i, start_idx, end_idx, start_box, end_box))
    return res

def refine_boxes(list_fids, list_boxes):
    """
    Interpolate missing boxes based on missing frame ids 
    """
    N = len(list_fids)
    res = []
    latest_idx = 0
    for i in range(N-1):
        if list_fids[i+1] - list_fids[i] > 1:
            if bb_intersection_over_union(list_boxes[i+1], list_boxes[i]) < 0.2:
                new_boxes = generate_boxes(
                    list_fids[i], list_fids[i+1], list_boxes[i], list_boxes[i+1]
                )
                
                res += (list_boxes[latest_idx : i+1] + new_boxes)
                latest_idx = i+1
                pass # Interpolate box
    
    res += list_boxes[latest_idx:]
    return res

def get_box_center(box: list):
    return (box[0]+box[2]/2, box[1]+box[3]/2)

def get_velocity(center_0, center_1):
    return (center_1[0]-center_0[0], center_1[1]-center_0[1])

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]

def xywh_to_xyxy_lst(boxes):
    new_boxes = []
    for box in boxes:
        new_boxes.append(xywh_to_xyxy(box))
    return new_boxes
