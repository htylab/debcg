"""Unified EEG-fMRI BCG processing utilities."""

__version__ = "0.1.0"

from importlib import import_module

from . import preprocessing  # noqa: F401


def __getattr__(name):
    if name in {"preprocessing", "deep", "brnet", "obs", "bcgnet", "dmh", "stats", "qrs"}:
        return import_module(f".{name}", __name__)
    raise AttributeError(name)

__all__ = ["preprocessing", "deep", "brnet", "obs", "bcgnet", "dmh", "stats", "qrs"]
