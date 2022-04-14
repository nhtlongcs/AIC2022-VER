# AIC2022-VER (WIP)

Before using this repo, please install the following packages by running the following commands:

```bash
conda install
pip install -e .
```

Extract data notebook: [colab](https://colab.research.google.com/drive/18Jmz-e4AvH1QAG_WVZqjlT3XgnPcWOMY)

Training Retrieval Model notebook: [colab](https://colab.research.google.com/drive/1o5g9fUndIFmHr-DYKczXq9q_-aiLQh-P)

Training Color Classification Model notebook: [colab](https://colab.research.google.com/drive/1KyywJ-nWrT-S8uRFfupinbpfltH0F7aA)

Training Vehicle Classification Model notebook: [colab](https://colab.research.google.com/drive/13eBvCeHoHKwTcoC9pXRQv9m_zmX0PFP5)

## Prepare data

Create a symbolic link to the data directory in the data directory of the project.

<!-- MAC OSX FAQ https://discussions.apple.com/thread/7423765 -->

```bash
$ cd /Users/your_short_username/path/to/where/you/want/to/put/the/symlink
$ ln -s /Volumes/HDD_name/path/to/where/you/are/storing/the/moved/files    symbolic_link_name_you_want_to_use
```

Ensure your data folder structure as same as our `data_sample`

```

$ ./tools/extract_vdo2frms_AIC.sh ./data/AIC22_Track2_NL_Retrieval/ ./data/meta/extracted_frames/
$ cp ./data/AIC22_Track2_NL_Retrieval/*.json ./data/meta/
$ ./tools/preproc.sh ./data/meta
$ ./tools/preproc_relation.sh ./data ./data/meta/test_tracks.json

$ ./tools/predict_srl.sh configs/srl/col_infer.yml artifacts/color/model.ckpt configs/srl/veh_infer.yml artifacts/vehicle/model.ckpt ./data/meta/relation/neighbor_tracks.json data/meta/extracted_frames ./out_test_neighbors/

$ ./tools/predict_srl.sh configs/srl/col_infer.yml artifacts/color/model.ckpt configs/srl/veh_infer.yml artifacts/vehicle/model.ckpt ./data/meta/test_tracks.json data/meta/extracted_frames ./out_test_tracks_colors/
```

With artifacts/ is the directory where you store the trained classification model.

For testing purpose, you can use the command `./tools/preproc.sh ./data_sample/meta`

Reading detail document of preprocessing part can be found in the [srl part](external/extraction/README.md) and [basic part](scripts/data/README.md) (adapted from hcmus team and alibaba team source code).

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

## Inference

We provide a simple inference script for inference purpose.

```bash
$ ./tools/infer.sh ./data/meta/
```

For detail, please take a look at `Predictor` class in `src/predictor.py`

## Contribution

### Development environment

If you want to contribute to this repo, please use the environment setup as below.

Pre-installation

Install conda according to the instructions on the homepage
Before installing the repo, we need to install the CUDA driver version >=10.2.

```

$ conda env create -f environment.yml
$ conda activate hcmus
$ pip install -r requirements.txt
$ pip install -e .

```

### Contribution guide

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
