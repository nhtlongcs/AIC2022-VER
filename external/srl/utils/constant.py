import os, json 
import os.path as osp

def dict_load(data_path: str):
    return json.load(open(data_path, 'r'))

SRL_DATA_DIR = 'external/extraction/configs' #osp.join(DATA_DIR, 'srl')
REFINED_TEST_TRACK_JSON = ''


# Attribute group information
VEHICLE_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'vehicle_vocabulary.json')
VEHICLE_OBJ_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'vehicle_vocabulary_object.json')

COLOR_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'color_vocabulary.json') 
ACTION_VOCAB_JSON = osp.join(SRL_DATA_DIR, 'action_vocabulary.json') 

VEHICLE_GROUP_JSON = osp.join(SRL_DATA_DIR, 'vehicle_group_v1.json') 
VEHICLE_GROUP_REP_JSON = VEHICLE_GROUP_JSON.replace('.json', '_rep.json')

COLOR_GROUP_JSON = osp.join(SRL_DATA_DIR, 'color_group_v1.json') 
COLOR_GROUP_REP_JSON = COLOR_GROUP_JSON.replace('.json', '_rep.json')

ACTION_GROUP_JSON = osp.join(SRL_DATA_DIR, 'action_group_v1.json') 
ACTION_GROUP_REP_JSON = ACTION_GROUP_JSON.replace('.json', '_rep.json')

# Preprocess result
TRAIN_SRL_JSON = osp.join(SAVE_DIR, 'result_train.json') 
TEST_SRL_JSON = osp.join(SAVE_DIR, 'result_test.json') 

VEHICLE_VOCAB = dict_load(VEHICLE_VOCAB_JSON)
VEHICLE_VOCAB_OBJ = dict_load(VEHICLE_OBJ_VOCAB_JSON) 

COLOR_VOCAB = dict_load(COLOR_VOCAB_JSON) 
ACTION_VOCAB = dict_load(ACTION_VOCAB_JSON) 
VEHICLE_GROUP_REP = dict_load(VEHICLE_GROUP_REP_JSON) #json.load(open(VEHICLE_GROUP_REP_JSON, 'r'))
LIST_REDUNDANT_VEHICLES = ['volvo', 'chevrolet', 'vehicle', 'car']

FOLLOW = "follow"
FOLLOW_BY = "followed by"

OPPOSITE = {
    FOLLOW: FOLLOW_BY,
    FOLLOW_BY: FOLLOW
}

HAS_FOLLOW = 2
NO_FOLLOW = -1
NO_CONCLUSION = 1

