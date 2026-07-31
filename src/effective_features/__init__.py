"""
EFFECTIVE-FEATURES - инструмент сравнительного анализа методов отбора признаков в задачах дистанционного зондирования Земли.
"""

from .config import ExperimentConfig, CLASS_NAMES, PALETTE, class_label
from .pipeline import run

__all__ = ['run', 'ExperimentConfig', 'CLASS_NAMES', 'PALETTE', 'class_label']
