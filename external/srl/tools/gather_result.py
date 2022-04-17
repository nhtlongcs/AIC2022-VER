import json
import os 
import os.path as osp
from tqdm import tqdm
import pandas as pd
import sys 

# sys.path.append('external/srl')

from external.srl.library.query import Query
from external.srl.utils.constant import (
    SAVE_DIR, 
    COLOR_GROUP_JSON, VEHICLE_GROUP_JSON, ACTION_GROUP_JSON,
    COLOR_GROUP_REP_JSON, VEHICLE_GROUP_REP_JSON, ACTION_GROUP_REP_JSON,
    TRAIN_SRL_JSON, TEST_SRL_JSON
)
from external.srl.utils.gather_utils import (
    get_label_info, get_label_vector, setup_info, 
    get_label_vector_with_split, get_rep_class
)
# from utils import prepare_dir, json_load, train_track_map, test_query_map
# from utils.file_handler import dict_save

## GLOBAL VARIABLES
train_save_dir = osp.join(SAVE_DIR, 'train_result')
test_save_dir = osp.join(SAVE_DIR, 'test_result')
train_csv_path = osp.join(SAVE_DIR, 'train_result_18July_23h02.csv')
# test_csv_path = osp.join(SAVE_DIR, 'test_result_19July_.csv')

# TEST_SRL_JSON = '/content/AI_City_2021/results/srl/srl_test_queries_normal.json'
test_csv_path = osp.join(SAVE_DIR, 'aic22_test_notOther_10Apr.csv')

# Test to deploy
TEST_SRL_JSON = '/content/AIC2022-VER/data_sample/meta/srl/srl_test_queries.json'
test_csv_path = '/content/test_srl.csv'

mode_save_dir = {
    'train': os.makedirs(train_save_dir, exist_ok=True), 
    'test': os.makedirs(test_save_dir, exist_ok=True), 
    'train_csv': train_csv_path,
    'test_csv': test_csv_path,
}

# key_map = {'train': train_track_map, 'test': test_query_map}
veh_info, col_info, act_info = {}, {}, {}
setup_info(veh_info, VEHICLE_GROUP_JSON)
setup_info(col_info, COLOR_GROUP_JSON)
setup_info(act_info, ACTION_GROUP_JSON)

## FUNCTIONS
def parse_result(srl_json: str, mode: str):
    srl_data = json.load(open(srl_json, 'r')) 
    list_ids = list(srl_data.keys())
    save_dir = mode_save_dir[mode]
    is_test = (mode == 'test')
    
    stat_dict = {
        'fail_query': [], 'svo_query': [],
    }
    list_res = []
    query_no_sub_veh = []
    query_no_sub_col = []
    
    for raw_key in tqdm(list_ids):
        query_dict = {}
        # new_key = key_map[mode][raw_key]
        query = Query(srl_data[raw_key], raw_key, None)
        srl_data[raw_key] = query.get_query_content_update()
        
        # save_path = osp.join(save_dir, f'{raw_key}.json')
        
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
    df_res.to_csv(mode_save_dir[f'{mode}_csv'], index=False)

    print(f'Query no subject vehicle: {len(query_no_sub_veh)} = {query_no_sub_veh}')
    print(f'Query no subject color: {len(query_no_sub_col)} = {query_no_sub_col}')
    pass


# ------------------------------------------------
def main():
    # print('Gather train SRL result')
    # parse_result(TRAIN_SRL_JSON, 'train')
    print('Gather test SRL result')
    parse_result(TEST_SRL_JSON, 'test')
    pass 

if __name__ == '__main__':
    main()
    pass
