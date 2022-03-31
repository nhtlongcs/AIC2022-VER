import json
import os.path as osp


class DataHelper(object):
    def __init__(self):
        pass

    def convert_json_train(self, json_data):
        # HARDCODE
        """
        Merge all the data into one list of tracklet
        """
        data = {}
        for key in json_data:
            data[key] = json_data[key]["nl"]
        return data

    def load_file(self, path):
        assert osp.exists(path), "File not found: {}".format(path)
        with open(path) as f:
            data = json.load(f)
        return data
