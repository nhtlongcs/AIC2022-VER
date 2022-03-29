import json
from pathlib import Path

ROOT = Path(__file__).parent
REFINED_TEST_TRACK_JSON = ""

VEHICLE_VOCAB_JSON = str(ROOT / "configs" / "vehicle_vocabulary.json")
COLOR_VOCAB_JSON = str(ROOT / "configs" / "color_vocabulary.json")
ACTION_VOCAB_JSON = str(ROOT / "configs" / "action_vocabulary.json")

VEHICLE_GROUP_JSON = str(ROOT / "configs" / "vehicle_group_v1.json")
VEHICLE_GROUP_REP_JSON = str(ROOT / VEHICLE_GROUP_JSON.replace(".json", "_rep.json"))
COLOR_GROUP_JSON = str(ROOT / "configs" / "color_group_v1.json")
ACTION_GROUP_JSON = str(ROOT / "configs" / "action_group_v1.json")

VEHICLE_VOCAB = json.load(open(VEHICLE_VOCAB_JSON, "r"))
COLOR_VOCAB = json.load(open(COLOR_VOCAB_JSON, "r"))
ACTION_VOCAB = json.load(open(ACTION_VOCAB_JSON, "r"))
VEHICLE_GROUP_REP = json.load(open(VEHICLE_GROUP_REP_JSON, "r"))
LIST_REDUNDANT_VEHICLES = ["volvo", "chevrolet", "vehicle", "car"]
