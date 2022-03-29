import os 
import os.path as osp

from utils import json_load

def setup_reverse_map(map_dict: dict):
    order_id_map = {}
    for k, v in map_dict.items():
        order_id_map[str(v)] = k 
    
    map_dict.update(order_id_map)
    return map_dict


COLUMN = 3
TOP_TO_SHOW = 15

# TITLE = 'Traffic Video Event Retrieval via Text Query using Vehicle Appearance and Motion Attributes'
TITLE = 'Traffic Video Event Retrieval via Text Query' 

TEST_TRACK_MAP_JSON = 'K:/Github/AIC2022-VER/UIUX/data/test_tracks_i2o.json'
TEST_QUERY_MAP_JSON = 'K:/Github/AIC2022-VER/UIUX/data/test_queries_i2o.json'

test_track_map = json_load(TEST_TRACK_MAP_JSON)
test_query_map = json_load(TEST_QUERY_MAP_JSON)

TEST_QUERY = 'K:/Github/AIC2022-VER/UIUX/data/test-queries.json'
TEST_TRACK = 'K:/Github/AIC2022-VER/UIUX/data/test-tracks.json'
test_query = json_load(TEST_QUERY)
video_dir = 'K:/Github/AIC2022-VER/UIUX/videos'

RESULT_DIR = 'K:/Github/AIC2022-VER/UIUX/results'

version_map = {
    'V0 (Retrieval Model)': osp.join(RESULT_DIR, 'v3_val_10e.json'),
    'V1 (V0 + vehcol)': osp.join(RESULT_DIR, 'v3_val_10e_veh-col.json'),
    'V2 (V0 + vehcol + action)': osp.join(RESULT_DIR, 'v3_val_10e_veh-col-act_19July.json'),
    'V3 (V0 + vehcol + relation)': osp.join(RESULT_DIR, 'v3_val_10e_veh-col-rel_19July.json'),
    'V4 (V0 + vehcol + action + relation)': osp.join(RESULT_DIR, 'v3_val_10e_act2_19July.json'),
}
