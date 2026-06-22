"""
compare.py — сравнение результатов нескольких запусков (по разным seed).

Назначение:
  Прогнав эксперимент с разными seed (разные случайные наборы патчей),
  мы собираем статистику устойчивости отбора признаков:
    - какие признаки выбираются ВСЕГДА (ядро отбора);
    - какие иногда (периферия);
    - какие почти никогда (шум).

  Это отвечает на вопрос: устойчив ли отбор к выбору обучающей выборки?
  Чем чаще признак попадает в отбор при разных seed, тем он надёжнее.

Результаты сохраняются в папку comparison/:
    summary.txt              — текстовая сводка с выводами
    freq_bhattacharyya.png   — частота выбора признаков (Бхаттачарья)
    freq_knn.png             — частота выбора признаков (kNN)
    stability_heatmap.png    — тепловая карта: признак × seed
"""

import os
from collections import Counter, OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .features import parse_feature_window


def _savefig(path, dpi=150):
    """Сохранение через буфер — безопасно к перехвату stdout."""
    import io
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches='tight', format='png')
    plt.close()
    buf.seek(0)
    with open(path, 'wb') as f:
        f.write(buf.read())


def aggregate_runs(runs):
    """
    runs — список словарей вида:
       {'seed': int,
        'bhattacharyya': [имена признаков в порядке отбора],
        'knn': [имена признаков в порядке отбора]}

    Порядок в списках = порядок Forward Selection (1-й = самый информативный).

    Возвращает:
      freq      — Counter частот по каждому методу
      per_seed  — {method: {seed: set(features)}}
      steps     — {method: {feature: [шаг_в_run1, шаг_в_run2, ...]}}
                  шаг = позиция в списке (1 = выбран первым)
    """
    n_runs = len(runs)
    methods = ['bhattacharyya', 'knn']

    freq = {m: Counter() for m in methods}
    per_seed = {m: {} for m in methods}
    steps = {m: {} for m in methods}   # {method: {feature: [steps...]}}

    for r in runs:
        seed = r['seed']
        for m in methods:
            feats = r.get(m, [])
            freq[m].update(feats)
            per_seed[m][seed] = set(feats)
            # Позиция признака в списке = шаг отбора (с 1)
            for pos, f in enumerate(feats, start=1):
                steps[m].setdefault(f, []).append(pos)

    return {'n_runs': n_runs, 'freq': freq, 'per_seed': per_seed,
            'steps': steps, 'seeds': [r['seed'] for r in runs]}


def _avg_step(steps_list):
    """Средний шаг выбора (по запускам, где признак был выбран)."""
    return sum(steps_list) / len(steps_list) if steps_list else None


def _classify(count, n_runs):
    """Классификация признака по частоте появления."""
    ratio = count / n_runs
    if ratio >= 0.999:
        return 'ядро'        # выбирается всегда
    elif ratio >= 0.5:
        return 'периферия'   # выбирается в большинстве запусков
    else:
        return 'шум'         # редко


def plot_frequency(freq_counter, n_runs, method_name, out_dir):
    """Бар-чарт: сколько раз каждый признак был выбран из n_runs запусков."""
    if not freq_counter:
        print(f"     Нет данных для {method_name}"); return
    items = freq_counter.most_common()
    names = [k for k, _ in items]
    counts = [v for _, v in items]

    # Цвет по устойчивости
    colors = []
    for c in counts:
        cls = _classify(c, n_runs)
        colors.append({'ядро': '#2A9D8F', 'периферия': '#FFB703',
                       'шум': '#E76F51'}[cls])

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 6))
    bars = ax.bar(range(len(names)), counts, color=colors,
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Сколько раз выбран (из {n_runs} запусков)', fontsize=11)
    ax.set_ylim(0, n_runs + 0.5)
    ax.set_yticks(range(n_runs + 1))
    ax.set_title(f'Частота выбора признаков — {method_name}\n'
                 f'(по {n_runs} запускам с разными seed)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.05, str(c),
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Линия «ядра» (выбран во всех запусках)
    ax.axhline(n_runs, color='#2A9D8F', ls='--', lw=1, alpha=0.6)

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#2A9D8F', label='Ядро (всегда)'),
        Patch(facecolor='#FFB703', label='Периферия (большинство)'),
        Patch(facecolor='#E76F51', label='Шум (редко)'),
    ]
    ax.legend(handles=legend, fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(out_dir, f'freq_{method_name}.png')
    _savefig(path)
    print(f"    График: {os.path.basename(path)}")


def plot_stability_heatmap(per_seed_method, seeds, all_features, method_name, out_dir):
    """
    Тепловая карта: строки — признаки, столбцы — seed.
    Ячейка закрашена, если признак выбран при данном seed.
    Сразу видно, какие признаки стабильны по всем столбцам.
    """
    if not all_features:
        print(f"     Нет данных для heatmap {method_name}"); return

    # Сортируем признаки по частоте (сверху самые стабильные)
    freq = Counter()
    for s in seeds:
        freq.update(per_seed_method.get(s, set()))
    features_sorted = [f for f, _ in freq.most_common()]

    matrix = np.zeros((len(features_sorted), len(seeds)))
    for j, s in enumerate(seeds):
        chosen = per_seed_method.get(s, set())
        for i, f in enumerate(features_sorted):
            matrix[i, j] = 1 if f in chosen else 0

    fig, ax = plt.subplots(figsize=(max(6, len(seeds) * 1.1),
                                    max(5, len(features_sorted) * 0.35)))
    cmap = matplotlib.colors.ListedColormap(['#f0f0f0', '#2A9D8F'])
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed={s}' for s in seeds], fontsize=9)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=9)
    ax.set_title(f'Устойчивость отбора — {method_name}\n'
                 f'(зелёный = признак выбран при этом seed)',
                 fontsize=12, fontweight='bold')

    # Сетка
    ax.set_xticks(np.arange(-.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(features_sorted), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    plt.tight_layout()
    path = os.path.join(out_dir, f'stability_{method_name}.png')
    _savefig(path)
    print(f"    График: {os.path.basename(path)}")


def write_summary(agg, out_dir, series_time=None, per_run_times=None, mode=None):
    """Текстовая сводка с выводами по устойчивости отбора."""
    path = os.path.join(out_dir, 'summary.txt')
    n = agg['n_runs']
    seeds = agg['seeds']
    lines = []
    lines.append("=" * 70)
    lines.append("  СВОДКА ПО СЕРИИ ЗАПУСКОВ (устойчивость отбора признаков)")
    lines.append("=" * 70)
    if mode:
        lines.append(f"  Режим:    {mode}")
    lines.append(f"  Запусков: {n}")
    lines.append(f"  Seeds:    {seeds}")
    if per_run_times:
        for s, t in per_run_times.items():
            lines.append(f"    seed={s}: {t:.1f} сек")
    if series_time is not None:
        lines.append(f"  Итого:    {series_time:.1f} сек")
    lines.append("")

    for method in ['bhattacharyya', 'knn']:
        freq = agg['freq'][method]
        lines.append("─" * 70)
        lines.append(f"  Критерий: {method}")
        lines.append("─" * 70)
        if not freq:
            lines.append("  (нет данных)")
            lines.append("")
            continue

        core   = [f for f, c in freq.items() if _classify(c, n) == 'ядро']
        periph = [f for f, c in freq.items() if _classify(c, n) == 'периферия']
        noise  = [f for f, c in freq.items() if _classify(c, n) == 'шум']

        lines.append(f"  ЯДРО (выбрано во всех {n} запусках) — {len(core)} шт:")
        for f in sorted(core, key=lambda x: -freq[x]):
            lines.append(f"     {f:<16s}  {freq[f]}/{n}")
        lines.append("")
        lines.append(f"  ПЕРИФЕРИЯ (в большинстве запусков) — {len(periph)} шт:")
        for f in sorted(periph, key=lambda x: -freq[x]):
            lines.append(f"     {f:<16s}  {freq[f]}/{n}")
        lines.append("")
        lines.append(f"  ШУМ (редко) — {len(noise)} шт:")
        for f in sorted(noise, key=lambda x: -freq[x]):
            lines.append(f"     {f:<16s}  {freq[f]}/{n}")
        lines.append("")

    # Сравнение ядер двух критериев — ТРИ УРОВНЯ согласованности
    freq_b = agg['freq']['bhattacharyya']
    freq_k = agg['freq']['knn']

    core_b = {f for f, c in freq_b.items() if _classify(c, n) == 'ядро'}
    core_k = {f for f, c in freq_k.items() if _classify(c, n) == 'ядро'}

    # Уровень 1 (строгий): оба выбрали ВО ВСЕХ запусках
    strict = core_b & core_k
    # Уровень 2 (практический): оба выбрали в БОЛЬШИНСТВЕ (>= половины)
    half = n / 2
    majority = {f for f in (set(freq_b) & set(freq_k))
                if freq_b[f] >= half and freq_k[f] >= half}
    # Уровень 3 (широкий): выбран обоими хотя бы раз
    ever = set(freq_b) & set(freq_k)

    lines.append("=" * 70)
    lines.append("  СОГЛАСОВАННОСТЬ КРИТЕРИЕВ (три уровня)")
    lines.append("=" * 70)
    lines.append(f"  Ядро Бхаттачарьи (всегда): {sorted(core_b)}")
    lines.append(f"  Ядро kNN (всегда):         {sorted(core_k)}")
    lines.append("")
    lines.append(f"  [1] СТРОГОЕ согласие — оба выбирают во ВСЕХ {n} запусках ({len(strict)}):")
    lines.append(f"      {sorted(strict)}")
    lines.append("")
    lines.append(f"  [2] СОГЛАСИЕ БОЛЬШИНСТВА — оба выбирают в >= {int(half)+ (1 if half%1 else 0)}/{n} запусках ({len(majority)}):")
    lines.append(f"      {sorted(majority)}")
    lines.append("")
    lines.append(f"  [3] ШИРОКОЕ согласие — выбран обоими хотя бы раз ({len(ever)}):")
    lines.append(f"      {sorted(ever)}")
    lines.append("")
    lines.append("  Интерпретация:")
    lines.append("    [1] нижняя граница — признаки, надёжные при любой выборке;")
    lines.append("    [2] практический набор — реальная согласованность методов;")
    lines.append("    [3] верхняя граница — все совместно отмеченные признаки.")
    lines.append("    Чем больше уровень [2], тем сильнее filter-метод (по формулам)")
    lines.append("    воспроизводит результат wrapper-метода (kNN) — что подтверждает")
    lines.append("    возможность отбора признаков без обучения классификатора.")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f"\n  Сводка сохранена: {path}")


def write_ranking_csv(agg, out_dir, sep=';'):
    """
    feature_ranking.csv — главная таблица ранжирования признаков.

    Колонки:
      feature        — имя признака
      group          — окно (3/5/7/9) или 'спектр'
      bhatta_count   — сколько раз выбран Бхаттачарьей (из N)
      bhatta_avg_step— средний шаг выбора (1=первым; пусто если не выбран)
      knn_count      — сколько раз выбран kNN
      knn_avg_step   — средний шаг выбора kNN
      category       — ядро / периферия / шум (по макс. из двух частот)

    Разделитель ';' — чтобы русский Excel открывал по двойному клику.
    Десятичная запятая — тоже для русского Excel.
    """
    import csv
    n = agg['n_runs']
    freq_b = agg['freq']['bhattacharyya']
    freq_k = agg['freq']['knn']
    steps_b = agg['steps']['bhattacharyya']
    steps_k = agg['steps']['knn']

    # Все признаки, выбранные хоть раз хоть одним методом
    all_feats = set(freq_b) | set(freq_k)

    def _grp(f):
        w = parse_feature_window(f)
        return f'окно_{w}' if w else 'спектр'

    def _num(x):
        """Число с запятой как десятичным разделителем (рус. Excel)."""
        if x is None:
            return ''
        return f"{x:.2f}".replace('.', ',')

    # Сортировка: сначала по суммарной частоте, потом по среднему шагу
    def _sort_key(f):
        total_count = freq_b.get(f, 0) + freq_k.get(f, 0)
        avg = _avg_step(steps_b.get(f, []) + steps_k.get(f, [])) or 99
        return (-total_count, avg)

    half = n / 2

    def _agreement(cb, ck):
        """
        Согласие методов по признаку:
          'оба'              — оба выбирают часто (>= половины запусков);
          'только Бхаттач.'  — берёт фильтр, kNN почти нет;
          'только kNN'       — берёт kNN, фильтр почти нет;
          'редко'            — оба берут редко.
        """
        b_often = cb >= half
        k_often = ck >= half
        if b_often and k_often:
            return 'оба'
        if b_often and not k_often:
            return 'только Бхаттач.'
        if k_often and not b_often:
            return 'только kNN'
        return 'редко'

    rows = []
    for f in sorted(all_feats, key=_sort_key):
        cb, ck = freq_b.get(f, 0), freq_k.get(f, 0)
        category = _classify(max(cb, ck), n)
        rows.append({
            'feature': f,
            'group': _grp(f),
            f'bhatta_count_из{n}': cb,
            'bhatta_avg_step': _num(_avg_step(steps_b.get(f, []))),
            f'knn_count_из{n}': ck,
            'knn_avg_step': _num(_avg_step(steps_k.get(f, []))),
            'category': category,
            'agreement': _agreement(cb, ck),
        })

    path = os.path.join(out_dir, 'feature_ranking.csv')
    cols = ['feature', 'group', f'bhatta_count_из{n}', 'bhatta_avg_step',
            f'knn_count_из{n}', 'knn_avg_step', 'category', 'agreement']
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=sep)
        w.writeheader()
        w.writerows(rows)
    print(f"    CSV: {os.path.basename(path)}  ({len(rows)} признаков)")


def write_per_seed_csv(agg, runs, out_dir, sep=';'):
    """
    per_seed.csv — что выбрано при каждом seed (сырые данные).
    Колонки: seed; method; step; feature
    """
    import csv
    path = os.path.join(out_dir, 'per_seed.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh, delimiter=sep)
        w.writerow(['seed', 'method', 'step', 'feature'])
        for r in runs:
            seed = r['seed']
            for method in ['bhattacharyya', 'knn']:
                for step, feat in enumerate(r.get(method, []), start=1):
                    w.writerow([seed, method, step, feat])
    print(f"    CSV: {os.path.basename(path)}")



def compare_runs(runs, comparison_dir, series_time=None, per_run_times=None, mode=None):
    """
    Главная функция сравнения.
    runs — список {'seed', 'bhattacharyya', 'knn'}.
    Строит графики, CSV-таблицы и сводку в comparison_dir.
    """
    os.makedirs(comparison_dir, exist_ok=True)
    agg = aggregate_runs(runs)
    seeds = agg['seeds']

    print("\n" + "─" * 60 + "\nПостроение сводных графиков\n" + "─" * 60)
    for method in ['bhattacharyya', 'knn']:
        plot_frequency(agg['freq'][method], agg['n_runs'], method, comparison_dir)
        all_feats = list(agg['freq'][method].keys())
        plot_stability_heatmap(agg['per_seed'][method], seeds, all_feats,
                               method, comparison_dir)

    print("\n" + "─" * 60 + "\nCSV-таблицы\n" + "─" * 60)
    write_ranking_csv(agg, comparison_dir)
    write_per_seed_csv(agg, runs, comparison_dir)

    print("\n" + "─" * 60 + "\nСводка\n" + "─" * 60)
    write_summary(agg, comparison_dir,
                  series_time=series_time, per_run_times=per_run_times, mode=mode)
    return agg