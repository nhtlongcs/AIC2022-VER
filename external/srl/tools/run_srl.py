import json
from srl_extractor import SRL
import os.path as osp
from utils.data_manager import RESULT_DIR, TRAIN_TRACK_JSON, TEST_QUERY_JSON

def main():
  # Running on train queries
  # Un-comment if you want to run on train data
  # srl_train = SRL(path=TRAIN_TRACK_JSON, filetype=1)
  # ans_train = srl_train.extract_data(srl_train.data)
  # f = open(osp.join(RESULT_DIR, "srl/result_train.json"), 'w')
  # json.dump(ans_train, f, indent=2)
  # f.close()

  # Running on test queries
  # Un-comment if you want to run on test data
  srl_test = SRL(path=TEST_QUERY_JSON, filetype=0)
  ans_test = srl_test.extract_data(srl_test.data)
  f = open(osp.join(RESULT_DIR, "srl/result_test_fix_thereis_fix_semi.json"), 'w')
  json.dump(ans_test, f, indent=2)
  f.close()

if __name__ == "__main__":
  main()