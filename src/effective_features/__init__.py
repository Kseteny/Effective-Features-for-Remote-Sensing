"""
EFFECTIVE-FEATURES — инструмент сравнительного анализа методов отбора
признаков в задачах дистанционного зондирования Земли.

Производственная практика / НИР, Сергеева К.С., группа 6301.

Быстрый старт:
    from effective_features import run, ExperimentConfig
    run()                                   # весь датасет, 41 признак
    run(ExperimentConfig(n_patches=8))      # быстрый прогон для отладки
"""

from .config import ExperimentConfig, CLASS_NAMES, PALETTE, class_label
from .pipeline import run

__all__ = ['run', 'ExperimentConfig', 'CLASS_NAMES', 'PALETTE', 'class_label']
