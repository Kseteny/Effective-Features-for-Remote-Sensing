"""
pipeline.py — главный пайплайн эксперимента.

Связывает все модули:
  config     — параметры
  features   — загрузка данных + 41 признак
  selectors  — расстояния + Forward Selection (реестр критериев)
  visualize  — графики

Точка входа: run(cfg).
"""

import os
import sys
import time
from datetime import timedelta
import numpy as np

from .config import ExperimentConfig, CLASS_NAMES
from .features import load_all_data, subsample_dataset, rebuild_feature_cube, parse_feature_window
from .selectors import (
    calculate_class_stats, compute_all_pairwise_distances, SELECTOR_REGISTRY,
)
from . import visualize as viz


class _Tee:
    """Дублирует вывод в консоль и в файл results.txt."""
    def __init__(self, filepath):
        self._file   = open(filepath, 'w', encoding='utf-8')
        self._stdout = sys.stdout
        sys.stdout   = self
    def write(self, data):
        self._stdout.write(data); self._file.write(data)
    def flush(self):
        self._stdout.flush(); self._file.flush()
    def close(self):
        sys.stdout = self._stdout; self._file.close()


def _fmt(seconds):
    """Человекочитаемое время: '1 мин 23 сек' или '45.2 сек'."""
    if seconds < 60:
        return f"{seconds:.1f} сек"
    return str(timedelta(seconds=int(seconds)))


def run(cfg: ExperimentConfig = None):
    """
    Полный пайплайн эксперимента.
    cfg=None → дефолтный конфиг (весь датасет, базовый набор 41 признак).
    """
    if cfg is None:
        cfg = ExperimentConfig()
    cfg.resolve_paths(__file__)

    log_path = os.path.join(cfg.results_dir, 'results.txt')
    tee = _Tee(log_path)

    t_start = time.perf_counter()
    timings = {}   # время каждого шага

    print("=" * 70)
    print("  EFFECTIVE-FEATURES — сравнительный анализ методов отбора признаков")
    print("=" * 70)
    print(f"  Окна:          {cfg.window_sizes}")
    print(f"  Спектральные:  {'да (9)' if cfg.use_spectral else 'нет'}")
    print(f"  Патчей:        {'весь датасет' if cfg.n_patches is None else cfg.n_patches}")
    print(f"  Лимит выборки: {cfg.max_pixels_total or 'нет'}")
    print(f"  max_features:  {cfg.max_features}")
    print(f"  Графики →      {cfg.output_dir}")

    # --- ШАГ 1: Загрузка ---
    print("\n" + "─" * 60 + "\nШАГ 1: Загрузка данных\n" + "─" * 60)
    _t = time.perf_counter()
    X_global, y_global, names = load_all_data(cfg)
    timings['Загрузка данных'] = time.perf_counter() - _t
    print(f"  ⏱  Загрузка: {_fmt(timings['Загрузка данных'])}")

    # --- ШАГ 2: Субдискретизация ---
    print("\n" + "─" * 60 + "\nШАГ 2: Субдискретизация\n" + "─" * 60)
    X_global, y_global = subsample_dataset(X_global, y_global, cfg)
    spec_names = [n for n in names if parse_feature_window(n) is None]
    text_names = [n for n in names if parse_feature_window(n) is not None]
    print(f"  Итоговая выборка: {len(X_global):,} пкс, {len(names)} признаков")
    print(f"  Спектральных: {len(spec_names)} | текстурных: {len(text_names)} | ВСЕГО: {len(names)}")

    unique_cls = np.unique(y_global)
    unique_cls = unique_cls[unique_cls > 0]
    print(f"  Классы: {unique_cls.tolist()}")

    dataset, mask = rebuild_feature_cube(X_global, y_global)

    # --- ШАГ 3: Расстояния ---
    print("\n" + "─" * 60 + "\nШАГ 3: Расстояния Махаланобиса и Бхаттачарьи\n" + "─" * 60)
    _t = time.perf_counter()
    stats = {}
    for cls in unique_cls:
        mv, cm = calculate_class_stats(dataset, mask, int(cls))
        if mv is not None:
            stats[int(cls)] = {'mean': mv, 'cov': cm}
    classes_list = sorted(stats.keys())
    df_maha, df_bhatt = compute_all_pairwise_distances(stats, classes_list)
    print("\n  D_B:\n" + df_bhatt.round(3).to_string())
    print("\n  D_M:\n" + df_maha.round(3).to_string())
    timings['Расстояния'] = time.perf_counter() - _t
    print(f"\n  ⏱  Расстояния: {_fmt(timings['Расстояния'])}")

    # --- ШАГ 4: Отбор по всем критериям из реестра ---
    print("\n" + "─" * 60 + "\nШАГ 4: Отбор признаков (все критерии)\n" + "─" * 60)
    _t = time.perf_counter()

    # Проверим доступность пары классов для Бхаттачарьи
    pair = cfg.bhatta_pair
    if not (pair[0] in unique_cls and pair[1] in unique_cls) and len(unique_cls) >= 2:
        cfg.bhatta_pair = (int(unique_cls[0]), int(unique_cls[1]))
        print(f"  Пара {pair} недоступна, взята {cfg.bhatta_pair}")

    results_by_method = {}
    for method_name, spec in SELECTOR_REGISTRY.items():
        print(f"\n  >>> Критерий: {method_name} ({spec['kind']})")
        _tm = time.perf_counter()
        func = spec['func']
        if spec['needs_target_classes']:
            sel, hist = func(dataset, mask, cfg,
                             target_classes=[int(c) for c in unique_cls])
        else:
            sel, hist = func(dataset, mask, cfg)
        elapsed = time.perf_counter() - _tm
        sel_names = [names[i] for i in sel]
        results_by_method[method_name] = {
            'indices': sel, 'history': hist, 'names': sel_names,
            'kind': spec['kind'], 'metric': spec['metric_name'],
            'time': elapsed,
        }
        print(f"  {method_name}: {sel_names}")
        print(f"  ⏱  {method_name}: {_fmt(elapsed)}")
    timings['Отбор признаков'] = time.perf_counter() - _t

    # --- ШАГ 5: Графики ---
    print("\n" + "─" * 60 + "\nШАГ 5: Построение рисунков\n" + "─" * 60)
    _t = time.perf_counter()
    sel_b_names = results_by_method.get('bhattacharyya', {}).get('names', [])
    sel_m_names = results_by_method.get('knn', {}).get('names', [])
    hist_b = results_by_method.get('bhattacharyya', {}).get('history', [])
    hist_m = results_by_method.get('knn', {}).get('history', [])

    viz.plot_feature_correlation(dataset, mask, names, cfg.output_dir)
    viz.plot_bhatta_heatmap(df_bhatt, cfg.output_dir)
    viz.plot_maha_heatmap(df_maha, cfg.output_dir)
    viz.plot_bhatta_forward(hist_b, sel_b_names, cfg.output_dir)
    viz.plot_knn_forward(hist_m, sel_m_names, cfg.output_dir)
    viz.plot_window_frequency(sel_b_names, sel_m_names, cfg.output_dir)
    viz.plot_criteria_agreement(sel_b_names, sel_m_names, names, cfg.output_dir)
    timings['Построение графиков'] = time.perf_counter() - _t

    # --- ИТОГ ---
    total = time.perf_counter() - t_start
    print("\n" + "=" * 70 + "\nИТОГОВЫЙ ОТЧЁТ\n" + "=" * 70)
    print(f"  Признаков всего : {len(names)} ({len(spec_names)} спектр. + {len(text_names)} текст.)")
    print(f"  Объём выборки   : {len(X_global):,}")
    print(f"  Классов         : {len(classes_list)}")
    for method_name, r in results_by_method.items():
        print(f"\n  {method_name} ({len(r['names'])}): {r['names']}")
    if sel_b_names and sel_m_names:
        common = set(sel_b_names) & set(sel_m_names)
        if common:
            print(f"\n  Согласованные ({len(common)}): {sorted(common)}")

    # --- ХРОНОМЕТРАЖ ---
    print("\n" + "─" * 60 + "\n⏱  ХРОНОМЕТРАЖ\n" + "─" * 60)
    for stage, sec in timings.items():
        bar = '█' * max(1, int(40 * sec / total))
        print(f"  {stage:<22s} {_fmt(sec):>12s}  {bar}")
    # отдельно — сравнение времени критериев (важно для выводов!)
    if all('time' in r for r in results_by_method.values()) and len(results_by_method) >= 2:
        print("\n  Сравнение критериев по времени:")
        for m, r in results_by_method.items():
            print(f"    {m:<16s} ({r['kind']:<8s}): {_fmt(r['time'])}")
    print(f"\n  ИТОГО: {_fmt(total)}")

    print("\n  Эксперимент завершён")
    print("=" * 70)
    tee.close()
    print(f"  Лог сохранён: {log_path}")

    return {
        'names': names, 'X': X_global, 'y': y_global,
        'df_maha': df_maha, 'df_bhatt': df_bhatt,
        'results': results_by_method,
        'timings': timings, 'total_time': total,
    }
