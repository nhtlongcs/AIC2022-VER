# SRL Handler Module

This module handles the following tasks:

- Read the output result from SRL Extraction step.
- Extract vehicle type, color, action label for training tracks
- Produce vehicle boxes used for Classifier module.

## Module organization

- `data`: defined attribute vocabulary and group used for the system.
- `results`: cropped images for each target objects and their labels. Our results are stored at [gdrive](https://drive.google.com/drive/folders/14Aho7AblVm6dHQzkTFTXGRpHFYuMmjNi?usp=sharing)

## Run extraction

```
python color_prep.py
python veh_prep.py
python action_prep.py
```
