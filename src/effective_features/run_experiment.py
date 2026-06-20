"""
run_experiment.py — точка запуска эксперимента с выбором режима и seed.

Запуск из папки src/:

  1) Режим + seed по умолчанию (42):
        python -m effective_features.run_experiment fast
        python -m effective_features.run_experiment research
        python -m effective_features.run_experiment full

  2) Режим + свой seed (другой набor патчей):
        python -m effective_features.run_experiment research 1
        python -m effective_features.run_experiment research 2
        python -m effective_features.run_experiment research 100

     Меняя seed, получаешь ДРУГОЙ случайный набор патчей.
     Если результат отбора при разных seed похож — он устойчив.
     Любые целые числа подойдут (удобно: 1, 2, 3, 4, 5).

  3) Без аргументов — интерактивное меню:
        python -m effective_features.run_experiment
"""

import sys
from dataclasses import replace
from effective_features import run, ExperimentConfig


# =========================================================================
# ГОТОВЫЕ РЕЖИМЫ (пресеты)
# =========================================================================
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


def choose_mode_interactive():
    """Показывает меню и считывает выбор режима + seed."""
    print("\n  Выберите режим запуска:\n")
    modes = list(PRESETS.keys())
    for i, m in enumerate(modes, 1):
        print(f"    {i}. {m:<10s} — {DESCRIPTIONS[m]}")
    print()
    while True:
        choice = input("  Введите номер (1-3) или название режима: ").strip().lower()
        if choice in PRESETS:
            mode = choice
            break
        if choice.isdigit() and 1 <= int(choice) <= len(modes):
            mode = modes[int(choice) - 1]
            break
        print("  Не понял. Введите 1, 2, 3 или fast / research / full.")

    seed_in = input("  Введите seed (Enter = 42, любое целое для другого набора): ").strip()
    seed = int(seed_in) if seed_in.isdigit() else 42
    return mode, seed


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        # Интерактивный режим
        mode, seed = choose_mode_interactive()
    else:
        mode = args[0].strip().lower()
        if mode not in PRESETS:
            print(f"  Неизвестный режим '{mode}'. Доступны: {', '.join(PRESETS)}")
            sys.exit(1)
        # Второй аргумент (необязательный) — seed
        if len(args) >= 2:
            if not args[1].lstrip('-').isdigit():
                print(f"  seed должен быть целым числом, а не '{args[1]}'")
                sys.exit(1)
            seed = int(args[1])
        else:
            seed = 42

    # Берём пресет и подставляем выбранный seed
    cfg = replace(PRESETS[mode], random_seed=seed)

    print(f"\n  Режим: '{mode}' — {DESCRIPTIONS[mode]}")
    print(f"  Seed:  {seed}" + ("  (стандартный)" if seed == 42 else "  (другой набор патчей)") + "\n")
    run(cfg)