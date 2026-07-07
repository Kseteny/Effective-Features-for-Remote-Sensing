"""
volume_experiment.py — эксперимент "точность от объёма датасета".

Запуск:
    python -m effective_features.volume_experiment
Результат:
    results/volume_experiment.csv — таблица (объём → точность/F1/время)
    В консоли — та же таблица + рекомендация по объёму стабилизации.
"""
import os
import csv
import time

import numpy as np

from .config import ExperimentConfig
from .features import load_all_data, subsample_dataset, rebuild_feature_cube
from .selectors import forward_selection_knn, evaluate_feature_set

# Целевые объёмы (число патчей при систематическом прореживании).
# Можно менять свободно — но не убирай несколько мелких точек (10-50),
# иначе кривую насыщения не разглядеть.
PATCH_COUNTS = [10, 25, 50, 75, 100, 150, 250, 400]

# Порог "стабилизации": если прирост accuracy между соседними объёмами
# меньше этого значения (в долях, не в процентах) — считаем, что кривая
# вышла на плато.
STABILIZATION_THRESHOLD = 0.005  # 0.5 п.п.


def _run_one(n_patches: int, seed: int) -> dict:
    """Один прогон: загрузка → kNN forward selection → оценка на контроле."""
    cfg = ExperimentConfig(
        use_thinning=True,
        thinning_target_patches=n_patches,
        random_seed=seed,
    )
    cfg.resolve_paths(__file__)

    t0 = time.perf_counter()
    X, y, names = load_all_data(cfg)
    X, y = subsample_dataset(X, y, cfg)
    dataset, mask = rebuild_feature_cube(X, y)

    unique_cls = np.unique(y)
    unique_cls = unique_cls[unique_cls > 0]
    target_classes = [int(c) for c in unique_cls]

    selected, _hist = forward_selection_knn(dataset, mask, cfg,
                                             target_classes=target_classes)
    eval_res = evaluate_feature_set(dataset, mask, selected, cfg,
                                     target_classes=target_classes)
    dt = time.perf_counter() - t0

    return {
        'n_patches': n_patches,
        'n_pixels': len(y),
        'n_classes': len(unique_cls),
        'n_features_selected': len(selected),
        'accuracy': round(eval_res['accuracy'], 4) if eval_res else None,
        'f1_macro': round(eval_res['f1_macro'], 4) if eval_res else None,
        'time_sec': round(dt, 1),
    }


def run_volume_experiment(patch_counts=None, seed=42, out_csv=None):
    patch_counts = patch_counts or PATCH_COUNTS
    rows = []

    for n in patch_counts:
        print(f"\n{'=' * 60}\nОбъём: {n} патчей (систематическое прореживание)\n{'=' * 60}")
        row = _run_one(n, seed)
        rows.append(row)
        print(f"  → {row['n_pixels']:,} пкс, {row['n_classes']} классов, "
              f"{row['n_features_selected']} признаков, "
              f"точность {row['accuracy']:.1%}, F1={row['f1_macro']:.3f}, "
              f"время {row['time_sec']:.0f} сек")

    # --- Поиск точки стабилизации ---
    print(f"\n{'=' * 60}\nСВОДНАЯ ТАБЛИЦА\n{'=' * 60}")
    print(f"{'Патчей':>8} {'Пикселей':>12} {'Признаков':>10} "
          f"{'Точность':>10} {'F1':>8} {'Время':>8}")
    for r in rows:
        print(f"{r['n_patches']:>8} {r['n_pixels']:>12,} {r['n_features_selected']:>10} "
              f"{r['accuracy']:>9.1%} {r['f1_macro']:>8.3f} {r['time_sec']:>7.0f}с")

    stabilized_at = None
    for i in range(1, len(rows)):
        remaining = rows[i:]
        if len(remaining) < 2:
            break
        accs = [r['accuracy'] for r in remaining if r['accuracy'] is not None]
        if len(accs) < 2:
            continue
        # "Стабилизация" — это когда ВЕСЬ последующий хвост укладывается
        # в узкий коридор вокруг текущей точки, а не просто соседняя
        # пара случайно совпала (иначе локальное плато посреди падающей
        # или растущей кривой ложно засчитывается как стабилизация).
        spread = max(accs) - min(accs)
        if spread < STABILIZATION_THRESHOLD * 2:
            stabilized_at = rows[i - 1]['n_patches']
            break

    trend_note = ""
    if len(rows) >= 3:
        accs_all = [r['accuracy'] for r in rows if r['accuracy'] is not None]
        deltas = [rows[i]['accuracy'] - rows[i - 1]['accuracy']
                  for i in range(1, len(rows))
                  if rows[i]['accuracy'] is not None and rows[i - 1]['accuracy'] is not None]
        net_change = accs_all[-1] - accs_all[0]
        n_down = sum(1 for d in deltas if d < 0)
        # Падающий тренд: итоговое изменение заметно отрицательное И
        # большинство шагов (не обязательно все — один локальный подъём
        # не должен всё перечёркивать) идут вниз.
        if net_change < -STABILIZATION_THRESHOLD * 2 and n_down >= len(deltas) * 0.6:
            trend_note = ("  ВНИМАНИЕ: похоже на устойчивое ПАДЕНИЕ точности с ростом "
                          "объёма, а не на стабилизацию — вероятно, при малом числе "
                          "патчей пропорции классов в выборке нерепрезентативны "
                          "(смещены в пользу редких/легко отделимых классов). "
                          "Стоит смотреть на этот тренд, а не искать точку насыщения.")

    print()
    if stabilized_at:
        print(f"  Похоже, точность стабилизируется начиная примерно с "
              f"{stabilized_at} патчей (весь последующий диапазон укладывается "
              f"в коридор {STABILIZATION_THRESHOLD*2:.1%}).")
    else:
        print("  Явной стабилизации в проверенном диапазоне не обнаружено — "
              "возможно, стоит добавить точки с большим числом патчей.")
    if trend_note:
        print(trend_note)

    # --- Сохранение CSV ---
    out_csv = out_csv or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'results', 'volume_experiment.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Сохранено: {out_csv}")

    return rows


if __name__ == '__main__':
    run_volume_experiment()