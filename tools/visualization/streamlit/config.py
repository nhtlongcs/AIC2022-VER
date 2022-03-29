import os
import os.path as osp

class StreamlitConfig:

    COLUMNS = 3
    TOP_TO_SHOW = 15
    TITLE = 'Traffic Video Event Retrieval via Text Query' 

    def __init__(self, query_json, video_dir, result_dir) -> None:
        self.result_dir = result_dir
        self.video_dir = video_dir
        self.query_json = query_json
        self.load_result_version(self.result_dir)

    def load_result_version(self, json_dir):
        json_files = os.listdir(json_dir)
        self.version_map = {
            json_name: osp.join(json_dir, json_name) for json_name in json_files
        }

