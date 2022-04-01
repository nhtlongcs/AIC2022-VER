# SRL Handler Module

This module handles the following tasks:

- Read the output result from SRL Extraction step.
- Extract vehicle type, color, action label for training tracks
- Produce vehicle boxes used for Classifier module.

## Module organization

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

## Run extraction

```

python color_prep.py
python veh_prep.py
python action_prep.py

```

```
