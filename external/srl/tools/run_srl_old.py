import nltk
nltk.download('wordnet')

import os 
import os.path as osp

from utils import TRAIN_TRACK_JSON, TEST_QUERY_JSON, dict_save
from srl.utils.constant import SAVE_DIR, TRAIN_SRL_JSON, TEST_SRL_JSON
from srl.library.srl_extractor import SRL

def run_mode(mode: str, data_path: str, save_path: str, filetype: int):
  srl_model = SRL(path=data_path, filetype=filetype)
  result = srl_model.extract_data(srl_model.data)
  # save_path = osp.join(SAVE_DIR, f'result_{mode}.json')
  dict_save(result, save_path)
  print(f'Run {mode}, save result to {save_path}')
  pass

def main():
  train_path = TRAIN_TRACK_JSON
  test_path = TEST_QUERY_JSON
  
  run_mode('train', train_path, TRAIN_SRL_JSON, 1)
  run_mode('test', test_path, TEST_SRL_JSON, 0)

  # # Running on train queries
  # srl_train = SRL(path=train_path, filetype=1)
  # ans_train = srl_train.extract_data(srl_train.data)
  # save_path = osp.join(SAVE_DIR, 'result_train.json')
  # dict_save(ans_train, save_path)
  
  # # Running on test queries
  # srl_test = SRL(path=test_path, filetype=0)
  # ans_test = srl_test.extract_data(srl_test.data)
  # f = open('./results/result_test.json', 'w')
  # json.dump(ans_test, f, indent=2)
  # f.close()

if __name__ == "__main__":
  main()