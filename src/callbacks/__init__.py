from registry import Registry
CALLBACKS_REGISTRY = Registry("CALLBACKS")

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


CALLBACKS_REGISTRY.register(EarlyStopping)
CALLBACKS_REGISTRY.register(ModelCheckpoint)