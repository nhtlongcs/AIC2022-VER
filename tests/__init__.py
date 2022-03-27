import os

from pytorch_lightning import seed_everything
from .helper import create_fake_data

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))
PACKAGE_ROOT = os.path.dirname(TEST_ROOT)
DATASETS_PATH = os.path.join(PACKAGE_ROOT, "data")
# generate a list of random seeds for each test
ROOT_SEED = 1234

if not (os.path.exists(os.path.join(DATASETS_PATH, "meta"))):
    create_fake_data()


def reset_seed():
    seed_everything()
