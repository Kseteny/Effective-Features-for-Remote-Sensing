"""
recompare.py — пересобрать сравнение из УЖЕ посчитанных папок,
не запуская эксперимент заново.

Читает results.txt из папок results/<mode>_seed<N>/, вытаскивает
отобранные признаки (Бхаттачарья и kNN) и строит сводное сравнение
в results/<mode>_comparison/ — за секунды, без пересчёта.

Запуск из папки src/:

  1) Конкретные сиды:
        python -m effective_features.recompare research 8 21 277 490 595

  2) Все найденные папки режима (автоопределение сидов):
        python -m effective_features.recompare research

  3) Интерактивно:
        python -m effective_features.recompare
"""

import os
import re
import sys
import glob

from effective_features.config import ExperimentConfig
from effective_features.compare import compare_runs


def _parse_results_txt(path):
    """
    Достаёт из results.txt списки признаков, отобранных каждым методом.
    Ищет строки вида:
        bhattacharyya (15): ['Mean_9', 'NDVI', ...]
        knn (9): ['Norm_B4', ...]
    Возвращает {'bhattacharyya': [...], 'knn': [...]}.
    """
    with open(path, encoding='utf-8') as f:
        text = f.read()

    result = {'bhattacharyya': [], 'knn': []}
    for method in ('bhattacharyya', 'knn'):
        # method (N): [ ... ]
        m = re.search(rf"{method}\s*\(\d+\):\s*\[(.*?)\]", text)
        if m:
            inside = m.group(1)
            # выдёргиваем имена в кавычках
            feats = re.findall(r"'([^']+)'", inside)
            result[method] = feats
    return result


def _find_seed_folders(project_root, mode):
    """Находит все папки results/<mode>_seed<N>/ и возвращает {seed: path}."""
    results_dir = os.path.join(project_root, 'results')
    pattern = os.path.join(results_dir, f'{mode}_seed*')
    found = {}
    for folder in glob.glob(pattern):
        name = os.path.basename(folder)
        m = re.match(rf'{mode}_seed(\d+)$', name)
        if m and os.path.isfile(os.path.join(folder, 'results.txt')):
            found[int(m.group(1))] = folder
    return found


def recompare(mode, seeds=None):
    """Собирает сравнение из готовых папок. seeds=None → все найденные."""
    # project_root определяем от расположения этого файла
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    all_folders = _find_seed_folders(project_root, mode)
    if not all_folders:
        print(f"  Не найдено папок results/{mode}_seed*/ с results.txt")
        return

    if seeds is None:
        seeds = sorted(all_folders.keys())
        print(f"  Найдены все сиды режима '{mode}': {seeds}")
    else:
        missing = [s for s in seeds if s not in all_folders]
        if missing:
            print(f"  Нет папок для сидов: {missing}")
            print(f"  Доступны: {sorted(all_folders.keys())}")
            seeds = [s for s in seeds if s in all_folders]
        if not seeds:
            print("  Нечего сравнивать.")
            return

    print(f"  Собираю сравнение по сидам: {seeds}")
    runs = []
    for seed in seeds:
        path = os.path.join(all_folders[seed], 'results.txt')
        parsed = _parse_results_txt(path)
        runs.append({
            'seed': seed,
            'bhattacharyya': parsed['bhattacharyya'],
            'knn': parsed['knn'],
        })
        print(f"    seed={seed}: Бхаттачарья {len(parsed['bhattacharyya'])}, "
              f"kNN {len(parsed['knn'])} признаков")

    comparison_dir = os.path.join(project_root, 'results', f'{mode}_comparison')
    print(f"\n  Сравнение → results/{mode}_comparison/")
    compare_runs(runs, comparison_dir, mode=mode)
    print(f"\n  Готово! (без пересчёта, только пересборка сравнения)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        mode = input("  Режим (fast/research/full): ").strip().lower() or 'research'
        raw = input("  Сиды через пробел (Enter = все найденные): ").strip()
        seeds = [int(s) for s in raw.split() if s.lstrip('-').isdigit()] or None
    else:
        mode = args[0].strip().lower()
        seeds = [int(s) for s in args[1:] if s.lstrip('-').isdigit()] or None

    recompare(mode, seeds)
