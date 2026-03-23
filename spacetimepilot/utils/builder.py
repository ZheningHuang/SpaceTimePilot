import copy
from .registry import Registry

PIPELINES = Registry("pipelines")
DATASETS = Registry("datasets")

def build_pipeline(cfg):
    """Build pipelines."""
    return PIPELINES.build(copy.deepcopy(cfg))

def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(copy.deepcopy(cfg)) 