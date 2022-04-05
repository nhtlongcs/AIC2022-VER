
AIC22_ROOT = "K:/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval"
AIC22_META_ROOT = "K:/Github/AIC2022-VER/data/meta"

TEST_TRACKS_JSON = f"{AIC22_ROOT}/test_tracks.json"
TRAIN_TRACKS_JSON = f"{AIC22_ROOT}/train_tracks.json"

TEST_AUX_TRACKS_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/test_neighbors.json"
TRAIN_AUX_TRACKS_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/train_neighbors.json"

TEST_AUX_TRACKS_MAPPING_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/test_neighbors_mapping.json"
TRAIN_AUX_TRACKS_MAPPING_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/train_neighbors_mapping.json"

TEST_AUX_TRACKS_MAPPING_REFINE_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/test_neighbors_mapping_refine.json"
TRAIN_AUX_TRACKS_MAPPING_REFINE_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/train_neighbors_mapping_refine.json"

TEST_RELATION_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/test_relation.json"
TRAIN_RELATION_JSON = f"{AIC22_META_ROOT}/relation/neightbor_tracks/json/train_relation.json"


TEST_CAM_IDS = [
        'S01/c001', 'S01/c002', 'S01/c003', 'S01/c004', 'S01/c005', 
        'S03/c010', 'S03/c011', 'S03/c012', 'S03/c013', 'S03/c014', 'S03/c015', 
        'S04/c016', 'S04/c017', 'S04/c018', 'S04/c019', 'S04/c020', 'S04/c021', 'S04/c022', 'S04/c023',
        'S04/c024', 'S04/c025', 'S04/c026', 'S04/c027', 'S04/c028', 'S04/c029', 'S04/c030', 'S04/c031',
        'S04/c032', 'S04/c033', 'S04/c034', 'S04/c035', 'S04/c036', 'S04/c037', 'S04/c038', 'S04/c039', 'S04/c040'
]

TRAIN_CAM_IDS = [
    'S02/c006', 'S02/c007', 'S02/c008', 'S02/c009', 
    'S05/c010', 'S05/c016',  'S05/c017',  'S05/c018', 'S05/c019', 'S05/c020', 
    'S05/c021', 'S05/c022',  'S05/c023', 'S05/c024', 'S05/c025',  'S05/c026', 
    'S05/c027', 'S05/c028',  'S05/c029', 'S05/c033', 'S05/c034',  'S05/c035', 'S05/c036'
]