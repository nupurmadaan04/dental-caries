"""
Root compatibility shim for dataset classes.
Directs imports to src.mlua.data.dataset.
"""
from src.mlua.data.dataset import TrainDataset, ValDataset

__all__ = ["TrainDataset", "ValDataset"]
