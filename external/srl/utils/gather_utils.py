import json
import numpy as np

def json_load(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)


def setup_info(data_info, data_json):
    num_classes, label_map = get_label_info(data_json)
    data_info['num_classes'] = num_classes
    data_info['label_map'] = label_map
    pass

def get_rep_class(group_rep_json: str, list_class: list):
    group_rep = json_load(group_rep_json)
    convert_map = {}
    for key in group_rep.keys():
        for val in group_rep[key]:
            convert_map[val] = key

    rep_class = []
    for class_name in list_class:
        if convert_map.get(class_name) is None:
            continue
        rep_class.append(convert_map[class_name])

    return rep_class

def get_label_info(label_group_json: str):
    label_group = json_load(label_group_json)
    num_classes = len(label_group.keys())

    id_map = {} #{'group-1': 0}
    for k in label_group.keys():
        i = int(k.split('-')[1]) - 1
        id_map[k] = i

    label_map = {} # {'suv': 2}
    for k in label_group.keys():
        i = id_map[k]
        for veh in label_group[k]:
            label_map[veh] = i 
    
    return num_classes, label_map

def get_label_vector(list_values, num_classes, label_map, is_test=True, use_fraction=False):
    y  = np.zeros(num_classes)
    flag = True # Check if exist at least one valid vehicle or not
    for val in list_values:
        if label_map.get(val) is None:
            # print(f'invalid value: {val}')
            continue
        flag = False
        if use_fraction:
            y[label_map[val]] += 1
        else:
            y[label_map[val]] = 1
    
    if flag:
        if is_test:
            return np.ones(num_classes).tolist()
        else:
            return None

    if use_fraction:
        y /= np.sum(y)

    return y.tolist()

def get_label_vector_with_split(list_values, num_classes, label_map):
    ans = []
    for val in list_values:
        y  = np.zeros(num_classes)
        if label_map.get(val) is None:
            y = np.ones(num_classes)
        else:
            y[label_map[val]] = 1

        ans.append(y.tolist())
    return ans