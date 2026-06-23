"""
run_experiment.py — единая точка запуска.

Из папки src/:
    python -m effective_features.run_experiment

Меню предложит выбрать:
    - одиночный запуск (один seed)
    - серию запусков (несколько seed) с автоматическим сравнением

Можно и сразу аргументами, без меню:
    python -m effective_features.run_experiment fast            # один прогон, seed=42
    python -m effective_features.run_experiment fast 7          # один прогон, seed=7
    python -m effective_features.run_experiment research 1 2 3  # серия (2+ seed → батч)
"""

import sys
from dataclasses import replace

from effective_features import run, ExperimentConfig
from effective_features.compare import compare_runs


PRESETS = {
    'fast':     ExperimentConfig(n_patches=10, max_pixels_total=120_000),
    'research': ExperimentConfig(n_patches=50),
    'full':     ExperimentConfig(),   # n_patches=None → весь датасет
}

DESCRIPTIONS = {
    'fast':     'Быстрый — 10 патчей, для отладки (пара минут)',
    'research': 'Исследовательский — 50 патчей, основной режим',
    'full':     'Полный — весь датасет, для финального результата (долго)',
}


def _choose_mode():
    """Меню выбора режима. В меню показываются fast и research;
    режим full остаётся доступен вводом названия вручную."""
    visible = ['fast', 'research']   # что показываем в списке
    print("\n  Выберите режим:\n")
    for i, m in enumerate(visible, 1):
        print(f"    {i}. {m:<10s} — {DESCRIPTIONS[m]}")
    print()
    while True:
        choice = input("  Режим (1, 2 или название): ").strip().lower()
        if choice in PRESETS:          # принимает и 'full', если введут вручную
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(visible):
            return visible[int(choice) - 1]
        print("  Введите 1, 2 или название режима (fast / research).")


def _choose_seeds():
    """
    Запрос seed(ов). Варианты ввода:
      • одно число        → одиночный запуск (напр.: 42)
      • несколько чисел   → серия со сравнением (напр.: 1 2 3 4 5)
      • 'rand' или 'r'    → случайные сиды (спросит, сколько)
      • Enter             → 42 (по умолчанию)

    Случайные сиды печатаются и сохраняются — эксперимент остаётся
    воспроизводимым (можно вписать те же числа повторно).
    """
    import random as _random
    print("\n  Введите seed(ы):")
    print("    • одно число   → одиночный запуск (напр.: 42)")
    print("    • через пробел → серия со сравнением (напр.: 1 2 3 4 5)")
    print("    • rand         → случайные сиды (спросит количество)")
    raw = input("  Seeds (Enter = 42): ").strip().lower()

    if not raw:
        return [42]

    # Случайные сиды
    if raw in ('rand', 'r', 'random', 'рандом'):
        cnt_in = input("  Сколько случайных сидов? (Enter = 5): ").strip()
        count = int(cnt_in) if cnt_in.isdigit() and int(cnt_in) > 0 else 5
        seeds = _random.sample(range(1, 10000), count)
        print(f"\n  Сгенерированы случайные сиды: {seeds}")
        print(f"  (запиши их, если захочешь повторить этот эксперимент)")
        return seeds

    seeds = [int(s) for s in raw.split() if s.lstrip('-').isdigit()]
    return seeds or [42]


def _single_run(mode, seed):
    """Одиночный запуск (результаты в output/ и results/, как обычно)."""
    cfg = replace(PRESETS[mode], random_seed=seed)
    print(f"\n  Режим: '{mode}' — {DESCRIPTIONS[mode]}")
    print(f"  Seed:  {seed}\n")
    run(cfg)


def _batch_run(mode, seeds):
    """Серия запусков по seed + автоматическое сравнение."""
    import os
    import time
    print("=" * 70)
    print(f"  СЕРИЯ ЗАПУСКОВ: режим '{mode}', seeds={seeds}")
    print("=" * 70)

    t_series = time.perf_counter()
    runs = []
    per_run_times = {}      # {seed: время запуска в секундах}
    project_root = None
    for idx, seed in enumerate(seeds, 1):
        tag = f"{mode}_seed{seed}"
        print(f"\n\n{'#' * 70}")
        print(f"#  ЗАПУСК {idx}/{len(seeds)} — seed={seed}  (папка: results/{tag}/)")
        print(f"{'#' * 70}")
        _t = time.perf_counter()
        cfg = replace(PRESETS[mode], random_seed=seed, run_tag=tag)
        res = run(cfg)
        per_run_times[seed] = time.perf_counter() - _t
        runs.append({
            'seed': seed,
            'bhattacharyya': res['results'].get('bhattacharyya', {}).get('names', []),
            'knn': res['results'].get('knn', {}).get('names', []),
            # эффективность набора каждого метода (для сводки по серии)
            'eval_bhattacharyya': res['results'].get('bhattacharyya', {}).get('eval'),
            'eval_knn': res['results'].get('knn', {}).get('eval'),
        })
        if project_root is None:
            project_root = cfg.project_root

    total_series = time.perf_counter() - t_series

    comparison_dir = os.path.join(project_root, 'results', f'{mode}_comparison')
    print(f"\n\n{'=' * 70}")
    print(f"  СРАВНЕНИЕ {len(seeds)} ЗАПУСКОВ → results/{mode}_comparison/")
    print(f"{'=' * 70}")
    # Передаём тайминги серии в сравнение — попадут в summary.txt
    compare_runs(runs, comparison_dir,
                 series_time=total_series, per_run_times=per_run_times,
                 mode=mode)
    print(f"\n  Готово! Результаты: results/{mode}_seed*/  и  results/{mode}_comparison/")


def main(mode, seeds):
    """Маршрутизация: 1 seed → одиночный, 2+ → батч со сравнением."""
    if len(seeds) == 1:
        _single_run(mode, seeds[0])
    else:
        _batch_run(mode, seeds)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        # Интерактив
        mode = _choose_mode()
        seeds = _choose_seeds()
    else:
        mode = args[0].strip().lower()
        if mode not in PRESETS:
            print(f"  Неизвестный режим '{mode}'. Доступны: {', '.join(PRESETS)}")
            sys.exit(1)
        if len(args) >= 2:
            # Поддержка: research rand [N]  → N случайных сидов
            if args[1].lower() in ('rand', 'r', 'random'):
                import random as _random
                count = int(args[2]) if len(args) >= 3 and args[2].isdigit() else 5
                seeds = _random.sample(range(1, 10000), count)
                print(f"  Сгенерированы случайные сиды: {seeds}")
                print(f"  (запиши их для воспроизводимости)")
            else:
                seeds = [int(s) for s in args[1:] if s.lstrip('-').isdigit()]
                if not seeds:
                    print("  Seeds должны быть целыми числами или 'rand'.")
                    sys.exit(1)
        else:
            seeds = [42]

    main(mode, seeds)
