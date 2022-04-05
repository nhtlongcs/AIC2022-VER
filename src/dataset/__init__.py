from registry import Registry
DATASET_REGISTRY = Registry("DATASET")

def default_loader(path):
    return Image.open(path).convert("RGB")

from .default import *
from .hcmus import *

