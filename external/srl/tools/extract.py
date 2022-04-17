import os, json, cv2, sys
import os.path as osp
import pandas as pd 
from tqdm import tqdm
import numpy as np 

from srl_handler.library.text.query import Query
from srl_handler.utils.constant import (
    TRAIN_SRL_JSON, TEST_SRL_JSON, TRAIN_TRACK_JSON, TEST_TRACK_JSON,
    VEHICLE_VOCAB, VEHICLE_GROUP_JSON, 
    COLOR_VOCAB, COLOR_GROUP_JSON, 
    ACTION_VOCAB_JSON, ACTION_GROUP_JSON
)
from srl.utils.common import (
    refine_list_colors, refine_list_subjects
)
from utils.data_manager import (
    test_query_map, train_track_map
)

srl_json = {'train': TRAIN_SRL_JSON, 'test': TEST_SRL_JSON}
key_map = {'train': train_track_map, 'test': test_query_map}
SAVE_DIR = '/home/ntphat/projects/AI_City_2021/srl_handler/results'

def parse(mode: str, mode_save_dir: str):
    mode_srl = json.load(open(srl_json[mode]))
    stat_dict = {
        'fail_query': [], 'svo_query': [],
    }
    list_res = []
    count = 0
    for old_key in tqdm(mode_srl):
        new_key = key_map[mode][old_key]
        json_save_path = osp.join(mode_save_dir, f'{new_key}.json')
        query_info = Query(mode_srl[old_key], query_id=old_key, query_order=new_key)
        query_sv = query_info.get_all_SV_info()
        query_svo = query_info.get_all_SVO_info()
        svo_data = []
        is_svo=False
        if len(query_info.subjects) == 0:
            stat_dict['fail_query'].append(old_key)
            
        if len(query_svo) > 0:
            for svo in query_svo:
                related_action = svo['V']
                related_object = svo['O']
                if related_action not in ['follow', 'followed']:
                    continue
                # print(related_object.vehicle)
                # print(related_object.combines)
                
                is_svo = True
                refined_veh = refine_list_subjects([related_object.vehicle], is_subject=False)[0]
                refined_col = None
                if len(related_object.combines) > 0:
                    refined_col = refine_list_colors(related_object.combines)
                    if len(refined_col) > 0:
                        refined_col = refined_col[0]
                svo_data.append((related_action, refined_col, refined_veh))
                count += 1

            if is_svo:
                stat_dict['svo_query'].append(key_map[mode][old_key])

        res_dict = {
            'query_id': old_key, 'query_order': new_key,
            'captions': query_info.get_list_captions_str(),
            'subject': query_info.subjects, 'color': query_info.colors, 
            'SV': [sv['V'] for sv in query_sv], 
            'SVO': svo_data,
        }
        list_res.append(res_dict)

        # if is_svo:
        #     print(res_dict)
        #     break
        with open(json_save_path, 'w') as f:
            json.dump(res_dict, f, indent=2) 


    print(f"{mode} has {len(stat_dict['svo_query'])} svo tracks")
    df_mode = pd.DataFrame(list_res)
    # df_mode.to_csv(osp.join(mode_save_dir, f'{mode}_srl.csv'), index=False)
    
    print(f'{mode} EDA:')
    for k in stat_dict:
        print(f'{k}: {len(stat_dict[k])}')
    print(f"fail queries: {stat_dict['fail_query']}")
    return df_mode, stat_dict

def main():
    for mode in ['train', 'test']:
        print('='*10 + f' Parse result in {mode} ' + '='*10)
        mode_save_dir = osp.join(SAVE_DIR, f'{mode}_srl')
        os.makedirs(mode_save_dir, exist_ok=True)
        df_mode, stat_dict = parse(mode, mode_save_dir)
        df_mode.to_csv(osp.join(SAVE_DIR, f'{mode}_srl.csv'), index=False)
        with open(osp.join(SAVE_DIR, f'{mode}_stat.json'), 'w') as f:
            json.dump(stat_dict, f, indent=2)

    pass

if __name__ == '__main__':
    main()