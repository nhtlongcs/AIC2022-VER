# AIC2022-VER

Before using this repo, please install the following packages by running the following commands:

```bash
conda install
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
$ python scripts/data/nlpaug_uts.py ./data/meta/train_tracks.json
$ python scripts/data/nlpaug_uts.py ./data/meta/test_tracks.json
```

Split data, train and test data into train and test data (Optional)
By running the following commands, you can split the data into train and test data in same folder.

```
$ python scripts/data/split.py ./data/meta/train_tracks.json
```

## Prepare data (HCMUS-team)

Extract train and test's queries into separated parts following the English PropBank Semantic Role Labeling rules.

```
$ python scripts/srl/extraction.py data/AIC22_Track2_NL_Retrieval/ data/meta
```

## Deployment

For deployment/training purpose, docker is an ready-to-use solution.

To build docker image:

```bash
$ cd <this-repo>
$ DOCKER_BUILDKIT=1 docker build -t aic22:latest .
```

To start docker container:

```bash
$ docker run --rm --name aic-t2 --gpus device=0 --shm-size 16G -it -v $(pwd)/:/home/workspace/src/ aic22:latest /bin/bash
```

With device is the GPU device number, and shm-size is the shared memory size (should be larger than the size of the model).

To attach to the container:

```bash
$ docker attach aic-t2
```

## Contribution guide

If you want to contribute to this repo, please follow steps below:

1. Fork your own version from this repository
1. Checkout to another branch, e.g. `fix-loss`, `add-feat`.
1. Make changes/Add features/Fix bugs
1. Add test cases in the `test` folder and run them to make sure they are all passed (see below)
1. Create and describe feature/bugfix in the PR description (or create new document)
1. Push the commit(s) to your own repository
1. Create a pull request on this repository

```
pip install pytest
python -m pytest tests/
```

Expected result:

```bash
============================== test session starts ===============================
platform darwin -- Python 3.7.12, pytest-7.1.1, pluggy-1.0.0
rootdir: /Users/nhtlong/workspace/aic/aic2022
collected 10 items

tests/test_args.py ...                                                     [ 30%]
tests/test_utils.py .                                                      [ 40%]
tests/uts/test_dataset.py .                                                [ 50%]
tests/uts/test_eval.py .                                                   [ 60%]
tests/uts/test_extractor.py ...                                            [ 90%]
tests/uts/test_model.py .                                                  [100%]
```
