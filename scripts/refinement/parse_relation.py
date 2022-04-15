import os 
import os.path as osp 
import json 
from tqdm import tqdm


def create_aic_format(rel_data: dict, neigh_data: dict, action_data: dict,save_dir: str = SAVE_DIR):
    stop_list = action_data['stop']
    turn_list = action_data['turn_left'] + action_data['turn_right']
    
    for query_id in tqdm(rel_data):
        query_data = rel_data[query_id]

        save_path = osp.join(save_dir, f'{query_id}.json')
        sub_id = 'sub'
        follow_objs = query_data['follow']
        followed_by_objs = query_data['followed_by']
        n_objs = len(follow_objs) + len(followed_by_objs)

        track_map = {} 
        for obj_id in follow_objs:
            track_map[obj_id] = neigh_data[obj_id]
        for obj_id in followed_by_objs:
            track_map[obj_id] = neigh_data[obj_id]
        
        track_data = {
            'subject': sub_id, 'n_objs': n_objs, 'follow': follow_objs, 'followed_by': followed_by_objs,
            'is_stop': (query_id in stop_list), 'is_turn': (query_id in turn_list),
            'track_map': track_map
        }
        with open(save_path, 'w') as f:
            json.dump(track_data, f, indent=2)
        

def main():
    
    # Input 
    TEST_RELATION = 'data/result/test_relation.json'
    TEST_NEIGHBOR = 'data/result/test_neighbors.json'
    TEST_ACTION = 'data/result/test_action_tuned_10Apr_1225/test_action_f1.json'

    # Output dir
    SAVE_DIR = 'data/result/test_relation_action_f1'
    os.makedirs(SAVE_DIR)

    relation_json = TEST_RELATION
    neighbor_json = TEST_NEIGHBOR
    action_json = TEST_ACTION
    
    action_data = json.load(open(action_json, 'r'))
    rel_data = json.load(open(relation_json, 'r'))
    neigh_data = json.load(open(neighbor_json, 'r'))

    create_aic_format(rel_data, neigh_data, action_data)
    pass 

if __name__ == '__main__':
    main()
