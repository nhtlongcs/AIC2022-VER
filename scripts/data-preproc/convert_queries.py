import json
from tqdm import tqdm

JSON_FILE = "K:/Github/AIC2022-VER/data/AIC22_Track2_NL_Retrieval/test_queries.json"
OUT_FILE = 'K:/Github/AIC2022-VER/data/queries/test_queries.json'

def extract_queries(json_file, out_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    track_ids = sorted(list(data.keys()))

    out_dict = {}
    for idx, track_id in enumerate(tqdm(track_ids)):
        out_dict[track_id] = data[track_id]['nl']

    with open(out_file, 'w') as f:
        json.dump(out_dict, f, indent = 4)

def extract_gts(json_file, out_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    track_ids = sorted(list(data.keys()))

    out_dict = {}
    for idx, track_id in enumerate(tqdm(track_ids)):
        out_dict[track_id] = [track_id]

    with open(out_file, 'w') as f:
        json.dump(out_dict, f, indent=4)

if __name__ == '__main__':
    extract_queries(JSON_FILE, OUT_FILE)
    # extract_gts(JSON_FILE, OUT_FILE)