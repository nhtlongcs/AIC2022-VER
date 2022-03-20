import os
import json
import random

RATIO = 0.2
SHUFFLE = True
random.seed(1812)

with open("data/AIC22_Track2_NL_Retrieval/train_tracks.json") as f:
    tracks_train = json.load(f)

keys = list(tracks_train.keys())
random.shuffle(keys)

train_data = dict()
val_data = dict()
val_len = int(len(keys) * RATIO)

for key in keys[:val_len]:
    val_data[key] = tracks_train[key]
for key in keys[val_len:]:
    train_data[key] = tracks_train[key]

with open("data/meta/train.json", "w") as f, open("data/meta/val.json", "w") as g:
    json.dump(train_data, f, indent=4)
    json.dump(val_data, g, indent=4)
