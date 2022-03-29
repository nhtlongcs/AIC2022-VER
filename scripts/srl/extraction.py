import json
from external.srl.extraction import SRL
import os.path as osp


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
    f = open(osp.join(RESULT_DIR, "result_test_fix_thereis_fix_semi.json"), "w")
    json.dump(ans_test, f, indent=2)
    f.close()


if __name__ == "__main__":
    TEST_QUERY_JSON = "data/AIC22_Track2_NL_Retrieval/test_queries.json"
    RESULT_DIR = "./"
    main()
