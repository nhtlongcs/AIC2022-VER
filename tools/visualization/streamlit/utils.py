import json 

def json_load(json_path: str):
    data = None
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


