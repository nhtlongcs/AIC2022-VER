from typing import List
import torch
from torch import nn
from . import METRIC_REGISTRY
from sklearn.metrics import f1_score
import numpy as np


@METRIC_REGISTRY.register()
class F1score:
    """
    F1 Score
    """

    def __init__(self, num_classes, average="weighted", label_key="labels", **kwargs):
        super().__init__(**kwargs)
        self.average = average
        self.label_key = label_key
        self.num_classes = num_classes
        self.threshold = kwargs.get("threshold", 0.5)
        self.reset()

    def update(self, preds, batch):
        """
        Perform calculation based on prediction and targets
        """
        preds = preds["logits"]
        targets = batch[self.label_key]

        preds = preds > self.threshold
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()

        self.preds += preds.numpy().tolist()
        self.targets += targets.numpy().tolist()

    def reset(self):
        self.targets = []
        self.preds = []

    def value(self):
        self.targets = np.concatenate(self.targets)
        self.preds = np.concatenate(self.preds)
        print(self.targets.shape)
        print(self.preds.shape)
        score = f1_score(self.targets, self.preds, average=self.average)
        return {f"f1": score}

