"""
run_experiment.py — точка запуска эксперимента с выбором режима из консоли.

Запуск из папки src/:

  1) С указанием режима сразу:
        python -m effective_features.run_experiment fast
        python -m effective_features.run_experiment research
        python -m effective_features.run_experiment full

  2) Без аргумента — покажет меню для выбора:
        python -m effective_features.run_experiment
"""

import sys
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
    """Показывает меню и считывает выбор пользователя."""
    print("\n  Выберите режим запуска:\n")
    modes = list(PRESETS.keys())
    for i, m in enumerate(modes, 1):
        print(f"    {i}. {m:<10s} — {DESCRIPTIONS[m]}")
    print()
    while True:
        choice = input("  Введите номер (1-3) или название режима: ").strip().lower()
        if choice in PRESETS:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(modes):
            return modes[int(choice) - 1]
        print("  Не понял. Введите 1, 2, 3 или fast / research / full.")


if __name__ == "__main__":
    # Режим из аргумента командной строки, либо интерактивный выбор
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()
        if mode not in PRESETS:
            print(f"  Неизвестный режим '{mode}'. Доступны: {', '.join(PRESETS)}")
            sys.exit(1)
    else:
        mode = choose_mode_interactive()

    print(f"\n  Режим запуска: '{mode}' — {DESCRIPTIONS[mode]}\n")
    run(PRESETS[mode])