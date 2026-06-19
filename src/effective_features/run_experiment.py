"""
run_experiment.py — точка запуска эксперимента.

Запуск из папки src/:
    python -m effective_features.run_experiment

Или просто раскомментируй нужный вариант ниже.
"""

# При запуске этого файла напрямую (python run_experiment.py)
# интерпретатор добавляет в sys.path папку с самим скриптом,
# из-за чего импорт пакета `effective_features` не находится.
# Добавим родительскую папку `src/` в sys.path, чтобы абсолютные
# импорты работали корректно в обоих режимах запуска.
import sys
from pathlib import Path

if __package__ is None:
    src_root = Path(__file__).resolve().parents[1]
    sys.path[0] = str(src_root)

from effective_features import run, ExperimentConfig


if __name__ == "__main__":
    # =====================================================================
    # ВЫБЕРИ КОНФИГУРАЦИЮ (раскомментируй одну строку)
    # =====================================================================

    # 1) Быстрый прогон для отладки — 8 патчей, ограничение выборки.
    #    Рекомендуется запустить ПЕРВЫМ, чтобы убедиться что всё читается.
    cfg = ExperimentConfig(n_patches=8, max_pixels_total=120_000)

    # 2) Полный прогон — весь датасет, без ограничений (долго!).
    # cfg = ExperimentConfig()

    # 3) Эксперимент с двумя окнами вместо четырёх.
    # cfg = ExperimentConfig(n_patches=8, window_sizes=(5, 9))

    # 4) Другая пара классов для Бхаттачарьи (например, лес vs луг).
    # cfg = ExperimentConfig(n_patches=8, bhatta_pair=(11, 14))

    # =====================================================================
    run(cfg)
