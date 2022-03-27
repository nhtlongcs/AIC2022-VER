# AIC2022-VER

Before using this repo, please install the following packages by running the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

Extract data notebook: [colab](https://colab.research.google.com/drive/18Jmz-e4AvH1QAG_WVZqjlT3XgnPcWOMY)

Training notebook: [colab](https://colab.research.google.com/drive/1o5g9fUndIFmHr-DYKczXq9q_-aiLQh-P)

## Prepare data

```
$ ./tools/extract_vdo2frms_AIC.sh  ./data/ ./data
$ python scripts/motion_map.py /content/AIC2022-VER/data/meta/
```

Generate augment data for training (Optional)

```
$ python -m spacy download en_core_web_sm
$ python scripts/nlpaug_uts.py ./data/meta/train_tracks.json
```
