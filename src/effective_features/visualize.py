"""
visualize.py - рисунки по итогам одного прогона.

  graph_01 - матрица корреляций признаков
  graph_02 - тепловая карта расстояний Бхаттачарьи
  graph_03 - тепловая карта расстояний Махаланобиса
  graph_04_<критерий> - кривая отбора, по одной на критерий
  graph_05 - вклад размеров окна
  graph_06 - согласованность критериев
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from .config import PALETTE, DEFAULT_COLORS
from .features import parse_feature_window


def _savefig(path, dpi=150):
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, bbox_inches='tight', format='png')
    plt.close()
    buf.seek(0)
    with open(path, 'wb') as f:
        f.write(buf.read())


# --------------------------------------------------------------------------- [1]
def plot_feature_correlation(dataset, mask, names, out_dir):
    """Матрица корреляций Пирсона; пары |r|>0.90 выделены золотой рамкой."""
    X = dataset.reshape(-1, dataset.shape[-1])[mask.flatten() > 0]
    if len(X) < 10:
        print("     Мало данных - рисунок 1 пропущен"); return
    corr = np.corrcoef(X.T)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.4), max(10, n * 0.35)))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Коэффициент корреляции Пирсона r')
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
    high = np.abs(corr) > 0.90
    for i in range(n):
        for j in range(n):
            if i != j and high[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor='gold', linewidth=1.2))
    ax.set_title(f'Матрица корреляций Пирсона ({n} признаков)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_01_feature_correlation.png')
    _savefig(path)
    print(f"    Рисунок 1: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [2]
def plot_bhatta_heatmap(df_bhatta, out_dir):
    """Тепловая карта попарных расстояний Бхаттачарьи."""
    arr = df_bhatta.values.copy().astype(float)
    np.fill_diagonal(arr, np.nan)
    data = pd.DataFrame(arr, index=df_bhatta.index, columns=df_bhatta.columns)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ccc',
                cbar_kws={'label': 'Расстояние Бхаттачарьи D_B'})
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_02_bhatta_heatmap.png')
    _savefig(path)
    print(f"    Рисунок 2: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [3]
def plot_maha_heatmap(df_maha, out_dir):
    """Тепловая карта попарных расстояний Махаланобиса."""
    arr = df_maha.values.copy().astype(float)
    np.fill_diagonal(arr, np.nan)
    data = pd.DataFrame(arr, index=df_maha.index, columns=df_maha.columns)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ccc',
                cbar_kws={'label': 'Расстояние Махаланобиса D_M'})
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_03_maha_heatmap.png')
    _savefig(path)
    print(f"    Рисунок 3: {os.path.basename(path)}")


# Цвета критериев: в criteria.py они заданы словами (для веб-интерфейса),
# здесь нужны обычные HEX для matplotlib.
CRIT_COLORS = {
    'forest': '#2A9D8F',
    'plum':   '#9B5DE5',
    'water':  '#3A86FF',
    'gold':   '#FFB703',
}


def _crit_color(criterion):
    return CRIT_COLORS.get(criterion.color, '#E63946')


# --------------------------------------------------------------------------- [4]
def plot_forward(history, sel_names, criterion, out_dir):
    """Кривая отбора для одного критерия: накопленное значение + прирост."""
    if not history:
        print(f"     Нет данных - кривая {criterion.id} пропущена"); return

    # kNN даёт долю правильных ответов, её удобнее показывать в процентах
    as_percent = criterion.id == 'knn'
    vals = [v * 100 for v in history] if as_percent else list(history)
    unit = 'точность, %' if as_percent else criterion.unit

    x = list(range(1, len(vals) + 1))
    gains = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    color = _crit_color(criterion)

    fig, ax1 = plt.subplots(figsize=(max(10, len(x) * 0.9), 5))
    ax2 = ax1.twinx()
    ax1.plot(x, vals, marker='o', lw=2.5, ms=8, color=color,
             markerfacecolor='white', markeredgewidth=2.5,
             label='накопленное', zorder=3)
    ax2.bar(x, gains, alpha=0.22, color=color, label='прирост')

    for xi, yi, nm in zip(x, vals, sel_names):
        ax1.annotate(nm, xy=(xi, yi), xytext=(xi, yi + max(vals) * 0.05),
                     ha='center', fontsize=8, rotation=30, color='#333',
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.8))

    if as_percent:
        ax1.set_ylim(0, 108)

    ax1.set_xlabel('Количество признаков (шаг отбора)', fontsize=11)
    ax1.set_ylabel(unit, fontsize=11)
    ax2.set_ylabel('прирост', color=color, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x); ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Отбор признаков: {criterion.name}',
                  fontsize=12, fontweight='bold')

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = os.path.join(out_dir, f'graph_04_forward_{criterion.id}.png')
    _savefig(path)
    print(f"    Рисунок 4 ({criterion.id}): {os.path.basename(path)}")


# --------------------------------------------------------------------------- [5]
def plot_window_frequency(selected, criteria, out_dir):
    """Сколько признаков каждого масштаба попало в набор - по каждому критерию.

    selected - {id критерия: [имена признаков]}
    """
    active = [c for c in criteria if selected.get(c.id)]
    if not active:
        print("     Нет данных - рисунок 5 пропущен"); return

    colors = {3: '#3A86FF', 5: '#2A9D8F', 7: '#FF9F1C', 9: '#E63946',
              None: '#888888'}
    labels = {3: 'Окно 3×3', 5: 'Окно 5×5', 7: 'Окно 7×7', 9: 'Окно 9×9',
              None: 'Спектральные'}

    fig, axes = plt.subplots(1, len(active), figsize=(6 * len(active), 5))
    if len(active) == 1:
        axes = [axes]

    for ax, c in zip(axes, active):
        counts = {w: 0 for w in [3, 5, 7, 9, None]}
        for nm in selected[c.id]:
            w = parse_feature_window(nm)
            counts[w] = counts.get(w, 0) + 1
        keys = [w for w in [3, 5, 7, 9, None] if counts[w] > 0]
        vals = [counts[w] for w in keys]

        bars = ax.bar(range(len(keys)), vals, color=[colors[w] for w in keys],
                      edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([labels[w] for w in keys], fontsize=10)
        ax.set_ylabel('Кол-во отобранных признаков', fontsize=11)
        ax.set_title(c.name, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(val), ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    plt.suptitle('Вклад текстурных масштабов в отобранный набор',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_05_window_frequency.png')
    _savefig(path)
    print(f"    Рисунок 5: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [6]
def plot_criteria_agreement(selected, criteria, names, out_dir):
    """Кто что выбрал: строки - критерии, столбцы - признаки.

    Закрашенная клетка = признак попал в набор этого критерия.
    Признаки сгруппированы по размеру окна.
    """
    active = [c for c in criteria if selected.get(c.id)]
    if not active:
        print("     Нет данных - рисунок 6 пропущен"); return

    # Порядок признаков: сначала спектральные, потом по возрастанию окна
    groups = {}
    for nm in names:
        w = parse_feature_window(nm)
        groups.setdefault(f'Окно {w}×{w}' if w else 'Спектральные', []).append(nm)
    order = ['Спектральные'] + [f'Окно {w}×{w}' for w in (3, 5, 7, 9)]
    ordered = [nm for g in order for nm in groups.get(g, [])]

    sets = {c.id: set(selected[c.id]) for c in active}
    matrix = np.array([[1 if nm in sets[c.id] else 0 for nm in ordered]
                       for c in active])

    fig, ax = plt.subplots(figsize=(max(14, len(ordered) * 0.32),
                                    2 + len(active) * 0.6))

    # Каждый критерий рисуем своим цветом: клетки закрашиваем построчно
    ax.imshow(np.zeros_like(matrix), aspect='auto',
              cmap=matplotlib.colors.ListedColormap(['#f0f0f0']))
    for i, c in enumerate(active):
        for j, val in enumerate(matrix[i]):
            if val:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color=_crit_color(c), zorder=2))

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels([c.name for c in active], fontsize=10)

    for i in range(1, len(ordered)):
        ax.axvline(i - 0.5, color='white', lw=0.6, zorder=3)
    for i in range(1, len(active)):
        ax.axhline(i - 0.5, color='white', lw=1.5, zorder=3)

    # Толстые линии между группами признаков + подписи групп
    pos = 0
    for g in order:
        feats = groups.get(g, [])
        if feats:
            if pos > 0:
                ax.axvline(pos - 0.5, color='black', lw=2, zorder=4)
            ax.text(pos + len(feats) / 2 - 0.5, -0.9, g, ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    transform=ax.get_xaxis_transform())
            pos += len(feats)

    # Сколько критериев сошлось на каждом признаке
    total = matrix.sum(axis=0)
    common = [nm for nm, t in zip(ordered, total) if t == len(active)]
    ax.set_title(f'Согласованность критериев отбора\n'
                 f'выбрано всеми ({len(common)}): '
                 f'{", ".join(common) if common else "-"}',
                 fontsize=12, fontweight='bold', pad=24)

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_06_criteria_agreement.png')
    _savefig(path)
    print(f"    Рисунок 6: {os.path.basename(path)}")
