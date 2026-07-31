"""
run_experiment.py - единая точка запуска.

Из папки src/:
    python -m effective_features.run_experiment          # спросит режим
    python -m effective_features.run_experiment fast     # сразу отладочный
    python -m effective_features.run_experiment thinned  # сразу основной

Патчи выбираются прореживанием - каждый k-й по порядку, плюс добор патчей
с редкими классами. Выборка одна и та же при каждом запуске, поэтому
результат воспроизводится без всяких сидов и серий прогонов.
"""

import sys

from effective_features import run, ExperimentConfig


PRESETS = {
    'fast':    ExperimentConfig(use_thinning=True, thinning_target_patches=10,
                                max_pixels_total=120_000),
    'thinned': ExperimentConfig(use_thinning=True, thinning_target_patches=150),
}

DESCRIPTIONS = {
    'fast':    'Отладочный - 10 патчей, проверить что ничего не сломалось',
    'thinned': 'Основной - ~150 патчей, все классы гарантированно',
}


def _choose_mode():
    """Меню выбора режима."""
    modes = list(PRESETS)
    print("\n  Выберите режим:\n")
    for i, m in enumerate(modes, 1):
        print(f"    {i}. {m:<10s} - {DESCRIPTIONS[m]}")
    print()
    while True:
        choice = input(f"  Режим (1-{len(modes)} или название): ").strip().lower()
        if choice in PRESETS:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(modes):
            return modes[int(choice) - 1]
        print(f"  Введите число 1-{len(modes)} или название режима "
              f"({', '.join(modes)}).")


def main(mode):
    print(f"\n  Режим: '{mode}' - {DESCRIPTIONS[mode]}\n")
    run(PRESETS[mode])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        mode = args[0].strip().lower()
        if mode not in PRESETS:
            print(f"  Неизвестный режим '{mode}'. Доступны: {', '.join(PRESETS)}")
            sys.exit(1)
    else:
        mode = _choose_mode()

    main(mode)
