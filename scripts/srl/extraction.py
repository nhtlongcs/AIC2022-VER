# Usage
import json
from external.srl.extraction import SRL
from pathlib import Path
import sys


def extract_textual_metadata(json_path, out_dir):
    assert Path(json_path).exists(), "json file not found"
    srl_test = SRL(path=json_path)
    filename = "srl_" + Path(json_path).stem
    out_path = Path(out_dir) / filename
    with open(str(out_path), "w") as f:
        ans_test = srl_test.extract_data(srl_test.data)
        json.dump(ans_test, f, indent=2)
        f.close()


def main():
    meta_data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    extract_textual_metadata(meta_data_dir / "train_tracks.json", out_dir)
    extract_textual_metadata(meta_data_dir / "test_queries.json", out_dir)


if __name__ == "__main__":
    main()
