import os, json 
import os.path as osp

from external.extraction.paths import (
    COLOR_GROUP_JSON, VEHICLE_GROUP_JSON, ACTION_GROUP_JSON,
    COLOR_GROUP_REP_JSON, VEHICLE_GROUP_REP_JSON, ACTION_GROUP_REP_JSON,
    VEHICLE_VOCAB, COLOR_VOCAB, ACTION_VOCAB, VEHICLE_GROUP_REP, VEHICLE_VOCAB_OBJ
)


LIST_REDUNDANT_VEHICLES = ['volvo', 'chevrolet', 'vehicle', 'car']
FOLLOW = "follow"
FOLLOW_BY = "followed by"

OPPOSITE = {
    FOLLOW: FOLLOW_BY,
    FOLLOW_BY: FOLLOW
}

HAS_FOLLOW = 2
NO_FOLLOW = -1
NO_CONCLUSION = 1

