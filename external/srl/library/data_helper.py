import json

class DataHelper(object):
  def __init__(self):
    pass

  def convert_json_train(self, json_data):
    data = {}
    for key in json_data:
      data[key] = json_data[key]['nl']
    return data
  
  def load_file(self, path):
    with open(path) as f:
      data = json.load(f)
    return data