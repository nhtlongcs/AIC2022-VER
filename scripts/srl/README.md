# SRL Handler Module

This module handles the following tasks:

- Read the output result from SRL Extraction step.
- Extract vehicle type, color, action label for training tracks
- Produce vehicle boxes used for Classifier module.

## Module organization

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> c094f38dbfe02f04d81e6adebdfc1fac140d4966
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

<<<<<<< HEAD
```
=======
- `data`: defined attribute vocabulary and group used for the system.
- `results`: cropped images for each target objects and their labels. Our results are stored at [gdrive](https://drive.google.com/drive/folders/14Aho7AblVm6dHQzkTFTXGRpHFYuMmjNi?usp=sharing)
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)
=======
- `data`: defined attribute vocabulary and group used for the system.
- `results`: cropped images for each target objects and their labels. Our results are stored at [gdrive](https://drive.google.com/drive/folders/14Aho7AblVm6dHQzkTFTXGRpHFYuMmjNi?usp=sharing)
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)

## Run extraction

```
<<<<<<< HEAD
<<<<<<< HEAD

=======
## Run extraction

```
>>>>>>> c094f38dbfe02f04d81e6adebdfc1fac140d4966
python color_prep.py
python veh_prep.py
python action_prep.py

```
<<<<<<< HEAD

=======
python color_prep.py
python veh_prep.py
python action_prep.py
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)
=======
python color_prep.py
python veh_prep.py
python action_prep.py
>>>>>>> fcf7e10 (update almost cleaned SRL scripts)
```
=======
>>>>>>> c094f38dbfe02f04d81e6adebdfc1fac140d4966
