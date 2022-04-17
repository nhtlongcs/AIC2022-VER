# SRL Extraction Module
This module handles the following task:
- Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.

## Module organization
- `data`: contains necessary annotation json files used for extraction. Can be downloaded at [gdrive](https://drive.google.com/drive/folders/1ksG1-S0l026LNlu3JZIc5ul-e1wM2l1F?usp=sharing)
- `results`: contains 2 result files of train and test after extraction. Our results are stored at [gdrive](https://drive.google.com/drive/folders/1-UZKNaNnx9YAAki5Ec3k9zlSsmCEcox_?usp=sharing)
- `notebook`: contains fundamental steps to run this module.

## Prepare
Install dependencies
```
pip install allennlp==2.0.0 -q
pip install allennlp-models==1.4.0 -q
pip install tqdm -q
pip install webcolors
pip install python-Levenshtein
```
## Run
```
cd './srl_extraction'
python main.py
```
The SRL parts obtained will be saved to `srl_extraction/results`