"""
compare.py - сводка по серии запусков с разными seed.

Показывает, насколько отбор устойчив: какие признаки выбираются всегда,
какие иногда, какие почти никогда. Работает с любым числом критериев.
Что складывается в comparison/:
    summary.txt          - текстовая сводка
    feature_ranking.csv  - таблица ранжирования
    per_seed.csv         - сырые данные по каждому seed
    freq_<критерий>.png, stability_<критерий>.png
"""

import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import criteria as crit
from .features import parse_feature_window
from .visualize import _crit_color


def _savefig(path, dpi=150):
    """Сохранение через буфер - безопасно к перехвату stdout."""
    import io
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches='tight', format='png')
    plt.close()
    buf.seek(0)
    with open(path, 'wb') as f:
        f.write(buf.read())


def _methods_in(runs):
    """Какие критерии реально встречаются в серии. Порядок - как в реестре."""
    present = set()
    for r in runs:
        present.update(r.get('selected', {}).keys())
    ordered = [c.id for c in crit.all_criteria() if c.id in present]
    # если попался критерий не из реестра - тоже не теряем
    return ordered + sorted(present - set(ordered))


def _name_of(method):
    c = crit.get(method)
    return c.name if c else method


def aggregate_runs(runs):
    """
    runs - список словарей:
       {'seed': int,
        'selected': {id критерия: [имена признаков в порядке отбора]},
        'evals':    {id критерия: результат оценки или None}}

    Возвращает частоты, шаги отбора и оценки, сгруппированные по критериям.
    """
    methods = _methods_in(runs)

    freq = {m: Counter() for m in methods}
    per_seed = {m: {} for m in methods}
    steps = {m: {} for m in methods}    # {критерий: {признак: [шаги]}}
    evals = {m: [] for m in methods}

    for r in runs:
        seed = r['seed']
        selected = r.get('selected', {})
        run_evals = r.get('evals', {}) or {}
        for m in methods:
            feats = selected.get(m, [])
            freq[m].update(feats)
            per_seed[m][seed] = set(feats)
            # позиция в списке = шаг отбора (1 = выбран первым)
            for pos, f in enumerate(feats, start=1):
                steps[m].setdefault(f, []).append(pos)
            if run_evals.get(m):
                evals[m].append(run_evals[m])

    return {'n_runs': len(runs), 'methods': methods, 'freq': freq,
            'per_seed': per_seed, 'steps': steps, 'evals': evals,
            'seeds': [r['seed'] for r in runs]}


def _avg_step(steps_list):
    """Средний шаг выбора по запускам, где признак был выбран."""
    return sum(steps_list) / len(steps_list) if steps_list else None


def _classify(count, n_runs):
    """Ядро - выбран всегда, периферия - в большинстве, шум - редко."""
    ratio = count / n_runs
    if ratio >= 0.999:
        return 'ядро'
    elif ratio >= 0.5:
        return 'периферия'
    return 'шум'


def _core(freq, n):
    return {f for f, c in freq.items() if _classify(c, n) == 'ядро'}


def plot_frequency(freq_counter, n_runs, method, out_dir):
    """Бар-чарт: сколько раз каждый признак был выбран из n_runs запусков."""
    if not freq_counter:
        print(f"     Нет данных для {method}"); return
    items = freq_counter.most_common()
    names = [k for k, _ in items]
    counts = [v for _, v in items]

    palette = {'ядро': '#2A9D8F', 'периферия': '#FFB703', 'шум': '#E76F51'}
    colors = [palette[_classify(c, n_runs)] for c in counts]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.45), 6))
    bars = ax.bar(range(len(names)), counts, color=colors,
                  edgecolor='white', linewidth=1.2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'Сколько раз выбран (из {n_runs} запусков)', fontsize=11)
    ax.set_ylim(0, n_runs + 0.5)
    ax.set_yticks(range(n_runs + 1))
    ax.set_title(f'Частота выбора признаков - {_name_of(method)}\n'
                 f'(по {n_runs} запускам с разными seed)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 0.05, str(c),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.axhline(n_runs, color='#2A9D8F', ls='--', lw=1, alpha=0.6)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2A9D8F', label='Ядро (всегда)'),
        Patch(facecolor='#FFB703', label='Периферия (большинство)'),
        Patch(facecolor='#E76F51', label='Шум (редко)'),
    ], fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(out_dir, f'freq_{method}.png')
    _savefig(path)
    print(f"    График: {os.path.basename(path)}")


def plot_stability_heatmap(per_seed_method, seeds, method, out_dir):
    """Строки - признаки, столбцы - seed. Клетка закрашена = признак выбран."""
    freq = Counter()
    for s in seeds:
        freq.update(per_seed_method.get(s, set()))
    if not freq:
        print(f"     Нет данных для heatmap {method}"); return

    features_sorted = [f for f, _ in freq.most_common()]
    matrix = np.array([[1 if f in per_seed_method.get(s, set()) else 0
                        for s in seeds] for f in features_sorted])

    c = crit.get(method)
    color = _crit_color(c) if c else '#2A9D8F'

    fig, ax = plt.subplots(figsize=(max(6, len(seeds) * 1.1),
                                    max(5, len(features_sorted) * 0.35)))
    cmap = matplotlib.colors.ListedColormap(['#f0f0f0', color])
    ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed={s}' for s in seeds], fontsize=9)
    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels(features_sorted, fontsize=9)
    ax.set_title(f'Устойчивость отбора - {_name_of(method)}\n'
                 f'(закрашено = признак выбран при этом seed)',
                 fontsize=12, fontweight='bold')

    ax.set_xticks(np.arange(-.5, len(seeds), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(features_sorted), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    plt.tight_layout()
    path = os.path.join(out_dir, f'stability_{method}.png')
    _savefig(path)
    print(f"    График: {os.path.basename(path)}")


def _fmt_time(seconds):
    """Время в человекочитаемом виде."""
    from datetime import timedelta
    if seconds is None:
        return '-'
    if seconds < 60:
        return f"{seconds:.1f} сек"
    return str(timedelta(seconds=int(seconds)))


def write_summary(agg, out_dir, series_time=None, per_run_times=None, mode=None):
    """Текстовая сводка по устойчивости отбора и согласию критериев."""
    path = os.path.join(out_dir, 'summary.txt')
    n = agg['n_runs']
    seeds = agg['seeds']
    methods = agg['methods']

    lines = []
    lines.append("=" * 70)
    lines.append("  СВОДКА ПО СЕРИИ ЗАПУСКОВ (устойчивость отбора признаков)")
    lines.append("=" * 70)
    lines.append(f"  Запусков:  {n}")
    lines.append(f"  Seeds:     {seeds}")
    lines.append(f"  Критериев: {len(methods)} ({', '.join(methods)})")
    if mode:
        lines.append(f"  Режим:     {mode}")
    lines.append("")

    if series_time is not None:
        lines.append("─" * 70)
        lines.append("  ВРЕМЯ ВЫПОЛНЕНИЯ СЕРИИ")
        lines.append("─" * 70)
        lines.append(f"  Всего по серии ({n} запусков): {_fmt_time(series_time)}")
        if per_run_times:
            avg = sum(per_run_times.values()) / len(per_run_times)
            lines.append(f"  В среднем на запуск:          {_fmt_time(avg)}")
            lines.append("")
            lines.append("  По каждому запуску:")
            for s in seeds:
                if s in per_run_times:
                    lines.append(f"     seed={s:<6d}  {_fmt_time(per_run_times[s])}")
        lines.append("")

    # Эффективность наборов, усреднённая по серии
    evals = agg.get('evals', {})
    if any(evals.get(m) for m in methods):
        lines.append("─" * 70)
        lines.append("  ЭФФЕКТИВНОСТЬ НАБОРОВ (среднее по серии, контрольная выборка)")
        lines.append("─" * 70)
        lines.append("  Классификатор kNN обучается на 70% пикселей, проверяется на 30%.")
        lines.append("")
        lines.append(f"  {'критерий':<22s} {'признаков':>9s} {'точность':>9s} "
                     f"{'ошибка':>8s} {'F1':>7s}")
        for m in methods:
            ev_list = evals.get(m, [])
            if not ev_list:
                continue
            k = len(ev_list)
            acc = sum(e['accuracy'] for e in ev_list) / k
            err = sum(e['error_rate'] for e in ev_list) / k
            f1 = sum(e['f1_macro'] for e in ev_list) / k
            nf = sum(e['n_features'] for e in ev_list) / k
            lines.append(f"  {m:<22s} {nf:>9.0f} {acc*100:>8.1f}% "
                         f"{err*100:>7.1f}% {f1:>7.3f}")
        lines.append("")

    # Ядро / периферия / шум по каждому критерию
    for m in methods:
        freq = agg['freq'][m]
        lines.append("─" * 70)
        lines.append(f"  Критерий: {m} - {_name_of(m)}")
        lines.append("─" * 70)
        if not freq:
            lines.append("  (нет данных)")
            lines.append("")
            continue
        for label in ('ядро', 'периферия', 'шум'):
            group = [f for f, c in freq.items() if _classify(c, n) == label]
            title = {'ядро': f'ЯДРО (выбрано во всех {n} запусках)',
                     'периферия': 'ПЕРИФЕРИЯ (в большинстве запусков)',
                     'шум': 'ШУМ (редко)'}[label]
            lines.append(f"  {title} - {len(group)} шт:")
            for f in sorted(group, key=lambda x: -freq[x]):
                lines.append(f"     {f:<16s}  {freq[f]}/{n}")
            lines.append("")

    # Согласие критериев
    half = n / 2
    cores = {m: _core(agg['freq'][m], n) for m in methods}
    chosen_ever = {m: set(agg['freq'][m]) for m in methods}
    often = {m: {f for f, c in agg['freq'][m].items() if c >= half} for m in methods}

    lines.append("=" * 70)
    lines.append("  СОГЛАСОВАННОСТЬ КРИТЕРИЕВ (три уровня)")
    lines.append("=" * 70)
    for m in methods:
        lines.append(f"  Ядро «{_name_of(m)}» ({len(cores[m])}): {sorted(cores[m])}")
    lines.append("")

    if len(methods) >= 2:
        strict = set.intersection(*cores.values())
        majority = set.intersection(*often.values())
        ever = set.intersection(*chosen_ever.values())

        lines.append(f"  [1] СТРОГОЕ согласие - все критерии выбирают "
                     f"во ВСЕХ {n} запусках ({len(strict)}):")
        lines.append(f"      {sorted(strict) if strict else '-'}")
        lines.append("")
        lines.append(f"  [2] СОГЛАСИЕ БОЛЬШИНСТВА - все критерии выбирают "
                     f"минимум в половине запусков ({len(majority)}):")
        lines.append(f"      {sorted(majority) if majority else '-'}")
        lines.append("")
        lines.append(f"  [3] ШИРОКОЕ согласие - выбран всеми хотя бы раз "
                     f"({len(ever)}):")
        lines.append(f"      {sorted(ever) if ever else '-'}")
        lines.append("")

        # Попарные пересечения ядер - видно, какие критерии ближе друг к другу
        if len(methods) > 2:
            lines.append("  Попарное согласие (пересечение ядер):")
            for i, a in enumerate(methods):
                for b in methods[i + 1:]:
                    both = cores[a] & cores[b]
                    lines.append(f"     {a} и {b}: {len(both)} - "
                                 f"{sorted(both) if both else '-'}")
            lines.append("")

        lines.append("  Интерпретация:")
        lines.append("    [1] нижняя граница - признаки, надёжные при любой выборке;")
        lines.append("    [2] практический набор - реальная согласованность методов;")
        lines.append("    [3] верхняя граница - все совместно отмеченные признаки.")
        lines.append("    Чем больше уровень [2], тем ближе быстрые критерии (по формулам)")
        lines.append("    воспроизводят результат kNN - а значит, отбор можно вести")
        lines.append("    без обучения классификатора.")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(text)
    print(f"\n  Сводка сохранена: {path}")


def write_ranking_csv(agg, out_dir, sep=';'):
    """
    feature_ranking.csv - главная таблица ранжирования признаков.

    На каждый критерий две колонки: сколько раз выбран и средний шаг выбора.
    Разделитель ';' и десятичная запятая - чтобы русский Excel открывал
    файл по двойному клику.
    """
    import csv
    n = agg['n_runs']
    methods = agg['methods']
    freq = agg['freq']
    steps = agg['steps']
    half = n / 2

    all_feats = set()
    for m in methods:
        all_feats |= set(freq[m])

    def _grp(f):
        w = parse_feature_window(f)
        return f'окно_{w}' if w else 'спектр'

    def _num(x):
        return '' if x is None else f"{x:.2f}".replace('.', ',')

    def _sort_key(f):
        total = sum(freq[m].get(f, 0) for m in methods)
        all_steps = [s for m in methods for s in steps[m].get(f, [])]
        return (-total, _avg_step(all_steps) or 99)

    def _agreement(f):
        """Сколько критериев берут признак часто (>= половины запусков)."""
        votes = [m for m in methods if freq[m].get(f, 0) >= half]
        if len(votes) == len(methods):
            return 'все'
        if not votes:
            return 'редко'
        return 'только ' + ', '.join(votes)

    cols = ['feature', 'group']
    for m in methods:
        cols += [f'{m}_count_из{n}', f'{m}_avg_step']
    cols += ['category', 'agreement']

    rows = []
    for f in sorted(all_feats, key=_sort_key):
        row = {'feature': f, 'group': _grp(f)}
        for m in methods:
            row[f'{m}_count_из{n}'] = freq[m].get(f, 0)
            row[f'{m}_avg_step'] = _num(_avg_step(steps[m].get(f, [])))
        row['category'] = _classify(max(freq[m].get(f, 0) for m in methods), n)
        row['agreement'] = _agreement(f)
        rows.append(row)

    path = os.path.join(out_dir, 'feature_ranking.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=sep)
        w.writeheader()
        w.writerows(rows)
    print(f"    CSV: {os.path.basename(path)}  ({len(rows)} признаков)")


def write_per_seed_csv(agg, runs, out_dir, sep=';'):
    """per_seed.csv - что выбрано при каждом seed. Колонки: seed;method;step;feature"""
    import csv
    path = os.path.join(out_dir, 'per_seed.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh, delimiter=sep)
        w.writerow(['seed', 'method', 'step', 'feature'])
        for r in runs:
            selected = r.get('selected', {})
            for method in agg['methods']:
                for step, feat in enumerate(selected.get(method, []), start=1):
                    w.writerow([r['seed'], method, step, feat])
    print(f"    CSV: {os.path.basename(path)}")


def compare_runs(runs, comparison_dir, series_time=None,
                 per_run_times=None, mode=None):
    """
    Главная функция сравнения.
    runs          - список {'seed', 'selected', 'evals'}.
    series_time   - общее время серии (сек), попадёт в summary.
    per_run_times - {seed: время запуска}.
    mode          - режим (fast/research/...), для summary.
    """
    os.makedirs(comparison_dir, exist_ok=True)
    agg = aggregate_runs(runs)
    seeds = agg['seeds']

    print("\n" + "─" * 60 + "\nПостроение сводных графиков\n" + "─" * 60)
    for method in agg['methods']:
        plot_frequency(agg['freq'][method], agg['n_runs'], method, comparison_dir)
        plot_stability_heatmap(agg['per_seed'][method], seeds, method,
                               comparison_dir)

    print("\n" + "─" * 60 + "\nCSV-таблицы\n" + "─" * 60)
    write_ranking_csv(agg, comparison_dir)
    write_per_seed_csv(agg, runs, comparison_dir)

    print("\n" + "─" * 60 + "\nСводка\n" + "─" * 60)
    write_summary(agg, comparison_dir, series_time=series_time,
                  per_run_times=per_run_times, mode=mode)
    return agg
