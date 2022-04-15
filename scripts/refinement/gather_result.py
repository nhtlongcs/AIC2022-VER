import sys
import json
import os 
import os.path as osp
from tqdm import tqdm
import pandas as pd

from external.extraction.heuristic.query import Query
from external.extraction.paths import (
    COLOR_GROUP_JSON, VEHICLE_GROUP_JSON, ACTION_GROUP_JSON,
    COLOR_GROUP_REP_JSON, ACTION_GROUP_REP_JSON,
)
from external.refinement.utils.gather import (
    get_label_vector, setup_info, 
    get_label_vector_with_split, get_rep_class
)


# key_map = {'train': train_track_map, 'test': test_query_map}
veh_info, col_info, act_info = {}, {}, {}
setup_info(veh_info, VEHICLE_GROUP_JSON)
setup_info(col_info, COLOR_GROUP_JSON)
setup_info(act_info, ACTION_GROUP_JSON)

## FUNCTIONS
def parse_result(srl_json: str, save_path: str, mode: str):
    srl_data = json.load(open(srl_json, 'r'))
    list_ids = list(srl_data.keys())
    is_test = (mode == 'test')
    
    stat_dict = {
        'fail_query': [], 'svo_query': [],
    }
    list_res = []
    query_no_sub_veh = []
    query_no_sub_col = []
    
    print(f'Starting ...')
    for raw_key in tqdm(list_ids):
        query_dict = {}
        # new_key = key_map[mode][raw_key]
        query = Query(srl_data[raw_key], raw_key)
        srl_data[raw_key] = query.get_query_content_update()
        
        # save_path = osp.join(save_dir, f'{new_key}.json')
        
        if len(query.subject_vehicle) == 0:
            stat_dict['fail_query'].append(raw_key)
        pass
        is_svo = False 
        if ('follow' in query.relation_actions) or ('followed' in query.relation_actions):
            is_svo = True

        
        is_sub_veh, is_sub_col = True, True
        subject_vehicle_label = get_label_vector(query.subject_vehicle, veh_info['num_classes'], veh_info['label_map'], is_test)
        subject_color_label = get_label_vector(query.subject_color, col_info['num_classes'], col_info['label_map'], is_test)
        
        if subject_vehicle_label is None:
            query_no_sub_veh.append(raw_key)
            is_sub_veh = False
        if subject_color_label is None:
            query_no_sub_col.append(raw_key)
            is_sub_col = False
        

        query_dict['query_id'] = raw_key
        # query_dict['query_order'] = new_key
        query_dict['captions'] = query.get_list_captions()
        query_dict['cleaned_captions'] = query.get_list_cleaned_captions()

        query_dict['subject_vehicle'] = query.subject_vehicle
        query_dict['subject_color'] = list(set(get_rep_class(COLOR_GROUP_REP_JSON, query.subject_color)))
        # query_dict['subject_color'] = query.subject_color
        
        query_dict['is_sub_veh'] = is_sub_veh
        query_dict['is_sub_col'] = is_sub_col

        # query_dict['action'] = query.actions
        query_dict['action'] = list(set(get_rep_class(ACTION_GROUP_REP_JSON, query.actions)))
        query_dict['relation_action'] = query.relation_actions
        query_dict['is_svo'] = is_svo

        query_dict['subject_vehicle_label'] = subject_vehicle_label
        query_dict['subject_color_label'] = subject_color_label
        query_dict['action_label'] = get_label_vector(query.actions, act_info['num_classes'], act_info['label_map'], is_test)

        query_dict['object_vehicle'] = query.object_vehicle
        query_dict['object_color'] = query.object_color
        query_dict['object_vehicle_label'] = get_label_vector_with_split(query.object_vehicle, veh_info['num_classes'], veh_info['label_map'])
        query_dict['object_color_label'] = get_label_vector_with_split(query.object_color, col_info['num_classes'], col_info['label_map'])
        
        query_dict['is_follow'] = query.is_follow
        
        # dict_save(query_dict, save_path)
        list_res.append(query_dict)        

    df_res = pd.DataFrame(list_res)
    df_res.to_csv(save_path, index=False)

    print(f'Query no subject vehicle: {len(query_no_sub_veh)} = {query_no_sub_veh}')
    print(f'Query no subject color: {len(query_no_sub_col)} = {query_no_sub_col}')
    pass


# ------------------------------------------------
def main():
    print('Gather test SRL result')

    # Input
    INP_SRL_JSON = sys.argv[1]
    SAVE_PATH = sys.argv[2]
    MODE = sys.argv[3]
    INP_SRL_JSON = 'data/result/srl/srl_test_queries.json'
    MODE = 'test'

    parse_result(INP_SRL_JSON, SAVE_PATH, MODE)

if __name__ == '__main__':
    main()
    pass
