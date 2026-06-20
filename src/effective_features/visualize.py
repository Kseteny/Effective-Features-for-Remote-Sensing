"""
visualize.py — построение итоговых рисунков эксперимента.

Рисунки:
  graph_01 — Матрица корреляций Пирсона признакового пространства
  graph_02 — Тепловая карта расстояний Бхаттачарьи
  graph_03 — Тепловая карта расстояний Махаланобиса
  graph_04 — Кривая Forward Selection (Бхаттачарья)
  graph_05 — Кривая Forward Selection (kNN Accuracy)
  graph_06 — Частота выбора признаков по размерам окна
  graph_07 — Согласованность критериев отбора
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
        print("     Мало данных — рисунок 1 пропущен"); return
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


# --------------------------------------------------------------------------- [4]
def plot_bhatta_forward(history, sel_names, out_dir):
    """Кривая Forward Selection по критерию Бхаттачарьи."""
    if not history:
        print("     Нет данных — рисунок 4 пропущен"); return
    x = list(range(1, len(history) + 1))
    gains = [history[0]] + [history[i] - history[i - 1] for i in range(1, len(history))]
    fig, ax1 = plt.subplots(figsize=(max(10, len(x) * 0.9), 5))
    ax2 = ax1.twinx()
    ax1.plot(x, history, marker='o', lw=2.5, ms=8, color='#E63946',
             markerfacecolor='white', markeredgewidth=2.5, label='D_B (накопл.)', zorder=3)
    ax2.bar(x, gains, alpha=0.22, color='#E63946', label='Прирост D_B')
    for xi, yi, nm in zip(x, history, sel_names):
        ax1.annotate(nm, xy=(xi, yi), xytext=(xi, yi + max(history) * 0.06),
                     ha='center', fontsize=8, rotation=30, color='#333',
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.8))
    ax1.set_xlabel('Количество признаков (шаг отбора)', fontsize=11)
    ax1.set_ylabel('Накопленное D_B', fontsize=11)
    ax2.set_ylabel('Прирост D_B', color='#E63946', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#E63946')
    ax1.set_xticks(x); ax1.grid(True, alpha=0.3)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc='lower right')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_04_forward_bhatta.png')
    _savefig(path)
    print(f"    Рисунок 4: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [5]
def plot_knn_forward(history, sel_names, out_dir):
    """Кривая Forward Selection по точности kNN."""
    if not history:
        print("     Нет данных — рисунок 5 пропущен"); return
    x = list(range(1, len(history) + 1))
    pct = [v * 100 for v in history]
    gains = [pct[0]] + [pct[i] - pct[i - 1] for i in range(1, len(pct))]
    fig, ax1 = plt.subplots(figsize=(max(10, len(x) * 0.9), 5))
    ax2 = ax1.twinx()
    ax1.plot(x, pct, marker='s', lw=2.5, ms=8, color='#2A9D8F',
             markerfacecolor='white', markeredgewidth=2.5, label='Accuracy (CV)', zorder=3)
    ax2.bar(x, gains, alpha=0.22, color='#2A9D8F', label='Прирост Acc')
    for xi, yi, nm in zip(x, pct, sel_names):
        ax1.annotate(nm, xy=(xi, yi), xytext=(xi, yi + max(pct) * 0.04),
                     ha='center', fontsize=8, rotation=30, color='#333',
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.8))
    ax1.axhspan(90, 105, alpha=0.06, color='green')
    ax1.axhspan(70, 90, alpha=0.06, color='yellow')
    ax1.axhspan(0, 70, alpha=0.06, color='red')
    ax1.set_ylim(0, 108)
    ax1.set_xlabel('Количество признаков (шаг отбора)', fontsize=11)
    ax1.set_ylabel('Точность классификации, %', fontsize=11)
    ax2.set_ylabel('Прирост точности, п.п.', color='#2A9D8F', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#2A9D8F')
    ax1.set_xticks(x); ax1.grid(True, alpha=0.3)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=9, loc='lower right')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_05_forward_knn.png')
    _savefig(path)
    print(f"    Рисунок 5: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [6]
def plot_window_frequency(sel_b_names, sel_m_names, out_dir):
    """Частота выбора признаков по размерам окна — какой масштаб информативнее."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {3: '#3A86FF', 5: '#2A9D8F', 7: '#FF9F1C', 9: '#E63946', None: '#888888'}
    labels = {3: 'Окно 3×3', 5: 'Окно 5×5', 7: 'Окно 7×7', 9: 'Окно 9×9', None: 'Спектральные'}
    for ax, sel_names, title in zip(
        axes, [sel_b_names, sel_m_names],
        ['Forward Selection: Бхаттачарья', 'Forward Selection: kNN']
    ):
        counts = {w: 0 for w in [3, 5, 7, 9, None]}
        for nm in sel_names:
            counts[parse_feature_window(nm)] = counts.get(parse_feature_window(nm), 0) + 1
        keys = [w for w in [3, 5, 7, 9, None] if counts[w] > 0]
        vals = [counts[w] for w in keys]
        cols = [colors[w] for w in keys]
        lbls = [labels[w] for w in keys]
        bars = ax.bar(range(len(keys)), vals, color=cols, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(lbls, fontsize=10)
        ax.set_ylabel('Кол-во отобранных признаков', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.suptitle('Вклад текстурных масштабов в отобранный набор',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_06_window_frequency.png')
    _savefig(path)
    print(f"    Рисунок 6: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [7]
def plot_criteria_agreement(sel_b_names, sel_m_names, names, out_dir):
    """Согласованность критериев: что выбрано обоими, что только одним."""
    set_b, set_m = set(sel_b_names), set(sel_m_names)

    def _code(nm):
        in_b, in_m = nm in set_b, nm in set_m
        if in_b and in_m: return 3
        if in_b: return 1
        if in_m: return 2
        return 0

    groups = {}
    for nm in names:
        w = parse_feature_window(nm)
        key = f'Окно {w}×{w}' if w else 'Спектральные'
        groups.setdefault(key, []).append(nm)
    order = ['Спектральные'] + [f'Окно {w}×{w}' for w in (3, 5, 7, 9)]
    ordered = []
    for g in order:
        ordered.extend(groups.get(g, []))

    codes = np.array([_code(nm) for nm in ordered]).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(max(14, len(ordered) * 0.32), 4))
    cmap = matplotlib.colors.ListedColormap(['#f0f0f0', '#E63946', '#2A9D8F', '#FFB703'])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(codes, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    pos = 0
    for g in order:
        feats = groups.get(g, [])
        if feats:
            if pos > 0:
                ax.axvline(pos - 0.5, color='black', lw=2)
            ax.text(pos + len(feats) / 2 - 0.5, -0.7, g, ha='center', va='top',
                    fontsize=9, fontweight='bold', transform=ax.get_xaxis_transform())
            pos += len(feats)
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='Не отобран'),
        Patch(facecolor='#E63946', edgecolor='gray', label='Только Бхаттачарья'),
        Patch(facecolor='#2A9D8F', edgecolor='gray', label='Только kNN'),
        Patch(facecolor='#FFB703', edgecolor='gray', label='Оба критерия'),
    ]
    ax.legend(handles=legend, loc='upper right', bbox_to_anchor=(1, 1.5), fontsize=9)
    ax.set_title('Согласованность критериев отбора (Бхаттачарья vs kNN)',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_07_criteria_agreement.png')
    _savefig(path)
    print(f"    Рисунок 7: {os.path.basename(path)}")
