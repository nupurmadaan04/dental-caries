"""
Root compatibility shim for dataloader samplers.
Directs imports to src.mlua.data.sampler.
"""
from src.mlua.data.sampler import TwoStreamBatchSampler

__all__ = ["TwoStreamBatchSampler"]
