"""Inference backends and configuration."""

from comp_model.inference.config import HierarchyStructure, InferenceConfig, PriorSpec
from comp_model.inference.dispatch import fit

__all__ = ["HierarchyStructure", "InferenceConfig", "PriorSpec", "fit"]
