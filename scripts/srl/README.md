# SRL Handler Module

This module handles the following tasks:

- Read the output result from SRL Extraction step.
- Extract vehicle type, color, action label for training tracks
- Produce vehicle boxes used for Classifier module.

## Module organization

<<<<<<< HEAD
```bash
|- this-repo
    |- data
    |- external
        |- extraction
            |- heuristic
            |- localize (not working yet)
            |- textual
    |- scripts
        |- srl
            |- veh-prep.py
            |- color-prep.py
            |- action-prep.py
            |- README.md
            |- ...
```

```
=======
- `data`: defined attribute vocabulary and group used for the system.
- `results`: cropped images for each target objects and their labels. Our results are stored at [gdrive](https://drive.google.com/drive/folders/14Aho7AblVm6dHQzkTFTXGRpHFYuMmjNi?usp=sharing)
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)

## Run extraction

```
<<<<<<< HEAD

python color_prep.py
python veh_prep.py
python action_prep.py

```

=======
python color_prep.py
python veh_prep.py
python action_prep.py
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)
```
