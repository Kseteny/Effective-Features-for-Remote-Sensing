"""
config.py — конфигурация эксперимента и справочники классов.

EFFECTIVE-FEATURES: инструмент сравнительного анализа методов отбора
признаков в задачах дистанционного зондирования Земли.
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt


# ===========================================================================
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ===========================================================================
@dataclass
class ExperimentConfig:
    """
    Все параметры эксперимента в одном месте. Меняя поля, можно запускать
    разные эксперименты, не трогая код пайплайна.

    Примеры:
        cfg = ExperimentConfig()                       # дефолт: весь датасет
        cfg = ExperimentConfig(n_patches=8)            # быстрый прогон (отладка)
        cfg = ExperimentConfig(window_sizes=(5, 9))    # только два окна
    """
    # --- Данные ---
    n_patches: Optional[int] = None
    # None = использовать ВЕСЬ датасет; число = ограничить N патчами
    max_pixels_total: Optional[int] = None
    # None = без ограничения общей выборки; число = субдискретизация
    random_seed: int = 42

    # --- Признаковое пространство (базовый набор = 41) ---
    window_sizes: Tuple[int, ...] = (3, 5, 7, 9)   # 4 окна → 32 текстурных
    use_spectral: bool = True                       # 9 спектральных

    # --- Forward Selection (общие) ---
    max_features: int = 15      # макс. длина отбираемого набора
    eps: float = 0.001          # порог останова по приросту критерия

    # --- Критерий Бхаттачарьи ---
    bhatta_pair: Tuple[int, int] = (2, 11)   # пара классов (застройка/лес)

    # --- Критерий kNN (wrapper) ---
    knn_k: int = 5                            # число соседей
    knn_cv: int = 5                           # число фолдов CV
    knn_max_samples: Optional[int] = 20_000   # лимит выборки только для kNN

    # --- Пути (если None — определяются автоматически) ---
    project_root: Optional[str] = None
    output_dir: Optional[str] = None
    results_dir: Optional[str] = None
    run_tag: Optional[str] = None
    # run_tag — метка запуска. Если задана (например 'seed7'), результаты
    # кладутся в отдельную папку results/<run_tag>/, графики туда же.
    # Нужно для серии запусков с разными seed (модуль compare).

    def resolve_paths(self, base_file: str):
        """
        Достраивает пути относительно структуры проекта.
        base_file — путь к вызывающему модулю (обычно pipeline.py).
        Структура: <root>/src/effective_features/<module>.py

        Если задан run_tag — все результаты (txt + графики) кладутся
        в отдельную папку results/<run_tag>/, чтобы запуски не перезатирались.
        """
        if self.project_root is None:
            self.project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(base_file))))

        if self.run_tag:
            # Изолированная папка под этот запуск
            base = os.path.join(self.project_root, 'results', self.run_tag)
            if self.results_dir is None:
                self.results_dir = base
            if self.output_dir is None:
                self.output_dir = base
        else:
            if self.output_dir is None:
                self.output_dir = os.path.join(self.project_root, 'output')
            if self.results_dir is None:
                self.results_dir = os.path.join(self.project_root, 'results')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        return self


# ===========================================================================
# СПРАВОЧНИКИ КЛАССОВ MultiSenGE (CORINE Land Cover)
# ===========================================================================
CLASS_NAMES = {
    1:  'Непрерывная городская застройка',
    2:  'Прерывистая городская застройка',
    3:  'Промышленные объекты',
    4:  'Дороги и ж/д пути',
    5:  'Портовые зоны',
    6:  'Аэропорты',
    7:  'Карьеры',
    8:  'Пляжи и дюны',
    9:  'Водно-болотные угодья',
    10: 'Торфяники',
    11: 'Широколиственные леса',
    12: 'Хвойные леса',
    13: 'Смешанные леса',
    14: 'Луга',
}

PALETTE = {
    1: '#E76F51', 2: '#E63946', 3: '#FF9F1C', 4: '#FFBE0B',
    5: '#8338EC', 6: '#3A86FF', 7: '#6D6875', 8: '#F4A261',
    9: '#48CAE4', 10: '#023E8A', 11: '#2A9D8F', 12: '#52B788',
    13: '#74C69D', 14: '#8AC926',
}
DEFAULT_COLORS = plt.cm.tab10.colors


def class_label(cls):
    """Короткая подпись класса для легенд графиков."""
    name = CLASS_NAMES.get(cls, f'Класс {cls}')
    return f"C{cls}: {name[:18]}{'…' if len(name) > 18 else ''}"