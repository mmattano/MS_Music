# ms_music/__init__.py

"""
MS Music Package
================

A Python package to generate audio from mass spectrometry data (.mzML files).
"""

from .sonifier import MSSonifier
from . import effects

__version__ = "0.1.0"

__all__ = ['MSSonifier', 'effects']

print(f"MS Music Package v{__version__} loaded.")