"""
=============================================================================
combined_module.py
=============================================================================
Курсовая работа: «Статистические методы отбора признаков при классификации
космических изображений (на примере датасета MultiSenGE)»

Объединённый модуль включает:
  1. fast_features.py          — признаковое пространство (текстура + спектр)
  2. loader.py                 — загрузка данных Sentinel-2
  3. evaluate_features.py      — расстояния Махаланобиса и Бхаттачарьи
  4. forward_selection_stats.py — Forward Selection по критерию Бхаттачарьи
  5. forward_selection_ml.py   — Forward Selection по точности kNN (5-fold CV)

Итоговые 10 рисунков (курсовая работа):
  [1]  graph_01 — Матрица корреляций Пирсона признакового пространства
  [2]  graph_02 — KDE-гистограммы распределений признаков по классам
  [3]  graph_03 — Тепловая карта попарных расстояний Бхаттачарьи
  [4]  graph_04 — Тепловая карта попарных расстояний Махаланобиса
  [5]  graph_05 — Кривая Forward Selection (критерий Бхаттачарьи)
  [6]  graph_06 — Кривая Forward Selection (kNN Accuracy, 5-fold CV)
  [7]  graph_07 — KDE-эллипсоиды рассеяния двух классов (Mean vs Rho_Avg)
  [8]  graph_08 — Гистограммы всех признаков для пары классов
  [9]  graph_09 — Box-plot нормализованных признаков по всем классам
  [10] graph_10 — Сравнительная таблица методов (comparison_table)
=============================================================================
"""

# ---------------------------------------------------------------------------
# Импорты
# ---------------------------------------------------------------------------
import os
import random
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter

try:
    import rasterio
    from scipy.ndimage import zoom
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠️  rasterio не установлен — запуск в демо-режиме (синтетические данные).")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Имена классов MultiSenGE (CORINE Land Cover)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    1:  'Непрерывная городская застройка',
    2:  'Прерывистая городская застройка',
    3:  'Промышленные объекты',
    4:  'Дороги и ж/д пути',
    5:  'Портовые зоны',
    6:  'Аэропорты',
    7:  'Карьеры',
    8:  'Пляжи и дюны',
    9:  'Водно-болотные угодья',
    10: 'Торфяники',
    11: 'Широколиственные леса',
    12: 'Хвойные леса',
    13: 'Смешанные леса',
    14: 'Луга',
}

PALETTE = {
    1: '#E76F51', 2: '#E63946', 3: '#FF9F1C', 4: '#FFBE0B',
    5: '#8338EC', 6: '#3A86FF', 7: '#6D6875', 8: '#F4A261',
    9: '#48CAE4', 10: '#023E8A', 11: '#2A9D8F', 12: '#52B788',
    13: '#74C69D', 14: '#8AC926',
}
DEFAULT_COLORS = plt.cm.tab10.colors

# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ MULTI-PATCH ЭКСПЕРИМЕНТА
# ---------------------------------------------------------------------------
N_PATCHES        = 8        # сколько случайных патчей брать
RANDOM_SEED      = 42       # для воспроизводимости
MAX_PIXELS_TOTAL = 120_000  # ограничение выборки (для скорости kNN)
WINDOW_SIZE      = 15       # окно текстурных признаков
USE_MULTISPECTRAL = True    # использовать ли спектральные индексы


def _label(cls):
    name = CLASS_NAMES.get(cls, f'Класс {cls}')
    return f"C{cls}: {name[:18]}{'…' if len(name)>18 else ''}"

# ===========================================================================
# ЧАСТЬ 1: ВЫЧИСЛЕНИЕ ПРИЗНАКОВ (fast_features.py)
# ===========================================================================

def get_fast_stats(image, window_size):
    """
    Локальное среднее, дисперсия и СКО за O(1) на пиксель.
    Использует integral image через uniform_filter.
    """
    img64   = image.astype(np.float64)
    mean    = uniform_filter(img64,      size=window_size, mode='mirror')
    sq_mean = uniform_filter(img64 ** 2, size=window_size, mode='mirror')
    var     = np.maximum(sq_mean - mean ** 2, 0.0)
    return mean, var, np.sqrt(var)


def calc_directional_rho(image, mean, var, window_size):
    """
    Направленные коэффициенты корреляции: 0°, 90°, 45°, 135°.
    Нормированная локальная ковариация пикселя со сдвинутым соседом.
    """
    eps  = 1e-10
    dirs = {'0': (0,1), '90': (1,0), '45': (1,1), '135': (1,-1)}
    rhos = {}
    for angle, (dy, dx) in dirs.items():
        shifted = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        f_adj   = uniform_filter(image * shifted, size=window_size, mode='mirror')
        rhos[angle] = np.clip((f_adj - mean**2) / (var + eps), -1.0, 1.0)
    return rhos


def compute_spectral_indices(img_10ch):
    """
    Спектральные индексы Sentinel-2.
    Порядок каналов: B2 B3 B4 B8 B5 B6 B7 B8A B11 B12
    """
    eps = 1e-8
    B2, B3, B4, B8 = img_10ch[0], img_10ch[1], img_10ch[2], img_10ch[3]
    B11             = img_10ch[8]
    total           = np.sum(img_10ch, axis=0)
    norm            = img_10ch / (total + eps)

    out = {
        'NDVI': ((B8 - B4) / (B8 + B4 + eps)).astype(np.float32),
        'NDWI': ((B3 - B8) / (B3 + B8 + eps)).astype(np.float32),
        'NDBI': ((B11- B8) / (B11+ B8 + eps)).astype(np.float32),
        'Total_Brightness': total.astype(np.float32),
    }
    for i, n in enumerate(['B2','B3','B4','B8','B5','B6','B7','B8A','B11','B12']):
        out[f'Norm_{n}'] = norm[i].astype(np.float32)
    return out


def extract_all_features(image, window_size=15, is_multispectral=False):
    """
    Полное признаковое пространство:
      - (опционально) спектральные индексы + нормированные каналы
      - локальное среднее Mean и СКО Std
      - 6 текстурных признаков: Rho_Avg, Rho_Range, Rho_0, Rho_90, Rho_45, Rho_135
    """
    fs = {}
    if is_multispectral and image.ndim == 3:
        print("  📡 Спектральные индексы (NDVI, NDWI, NDBI)...")
        fs.update(compute_spectral_indices(image))
        gray = np.mean(image, axis=0).astype(np.float32)
    else:
        gray = image.astype(np.float32)
        fs['Original'] = gray

    print(f"  📐 Локальные статистики (окно {window_size}×{window_size})...")
    mean, var, std = get_fast_stats(gray, window_size)
    fs['Mean'] = mean.astype(np.float32)
    fs['Std']  = std.astype(np.float32)

    print("  🧵 Направленные корреляции (0°, 90°, 45°, 135°)...")
    rhos = calc_directional_rho(gray, mean, var, window_size)
    fs['Rho_Avg']   = ((rhos['0']+rhos['90']+rhos['45']+rhos['135'])/4).astype(np.float32)
    fs['Rho_Range'] = (
        np.maximum.reduce(list(rhos.values())) -
        np.minimum.reduce(list(rhos.values()))
    ).astype(np.float32)
    for a in ('0','90','45','135'):
        fs[f'Rho_{a}'] = rhos[a].astype(np.float32)
    return fs


def make_feature_sandwich(fd):
    """dict → (H,W,C) float32 + список имён каналов."""
    names = list(fd.keys())
    return np.stack([fd[n].astype(np.float32) for n in names], axis=-1), names

# ===========================================================================
# ЧАСТЬ 2: ЗАГРУЗКА ДАННЫХ (loader.py)
# ===========================================================================

SENTINEL2_CHANNELS = {
    'B2':1,'B3':2,'B4':3,'B8':4,'B5':5,'B6':6,'B7':7,'B8A':8,'B11':9,'B12':10
}


def load_pair(s2_name, gr_name, data_dir=None):
    """Загружает (10,H,W) float32 снимок и (H,W) uint8 маску."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio не установлен")
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    with rasterio.open(os.path.join(data_dir,'s2_pref', s2_name.strip())) as s:
        image = s.read().astype(np.float32)
    with rasterio.open(os.path.join(data_dir,'ground_reference', gr_name.strip())) as s:
        mask = s.read(1).astype(np.uint8)
    return image, mask


def get_file_lists(lists_dir=None):
    """Читает out_s2_pref.txt / out_gr_pref.txt."""
    if lists_dir is None:
        lists_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','lists')
    with open(os.path.join(lists_dir,'out_s2_pref.txt')) as f:
        s2 = [l.strip() for l in f if l.strip()]
    with open(os.path.join(lists_dir,'out_gr_pref.txt')) as f:
        gr = [l.strip() for l in f if l.strip()]
    return s2, gr


def select_random_pairs(n_patches=N_PATCHES, seed=RANDOM_SEED):
    """
    Случайным образом выбирает N патчей из датасета MultiSenGE.
    Возвращает список кортежей (s2_filename, gr_filename).
    """
    s2_list, gr_list = get_file_lists()

    if len(s2_list) != len(gr_list):
        raise ValueError("Количество снимков и масок не совпадает")

    pairs = list(zip(s2_list, gr_list))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    selected = pairs[:n_patches]

    print(f"\n🎲 Выбрано случайных патчей: {len(selected)}")
    for i, (s2, gr) in enumerate(selected, 1):
        print(f"  {i}. {s2}")

    return selected


def build_global_dataset(feature_cube, mask):
    """
    Преобразует (H,W,C) → табличную выборку валидных (непустых) пикселей.

    Параметры
    ---------
    feature_cube : ndarray (H, W, C)
    mask         : ndarray (H, W), 0 = фон

    Возвращает
    ----------
    X : ndarray (N, C)  — признаки
    y : ndarray (N,)    — метки классов
    """
    h, w, c = feature_cube.shape
    X = feature_cube.reshape(-1, c)
    y = mask.flatten()

    valid = y > 0
    return X[valid], y[valid]


def subsample_dataset(X, y, max_samples=MAX_PIXELS_TOTAL, seed=RANDOM_SEED):
    """
    Случайное ограничение размера выборки для ускорения kNN и Cross-Validation.
    """
    if len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]


def rebuild_feature_cube(X, y):
    """
    Временно восстанавливает псевдо-куб (N,1,C) и маску (N,1) из плоской
    выборки — для совместимости с функциями визуализации, ожидающими 3D-массив.
    """
    n, c = X.shape
    dataset = X[:, np.newaxis, :]
    mask    = y.reshape(n, 1)
    return dataset, mask


def auto_find_data(project_root=None):
    """Ищет первую пару .tif в стандартной структуре проекта."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = mask_path = None
    for d, key in [(os.path.join(project_root,'data','s2_pref'),     'img'),
                   (os.path.join(project_root,'data','ground_reference'),'mask')]:
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.tif','.tiff')):
                if key == 'img'  and img_path  is None: img_path  = os.path.join(d,f)
                if key == 'mask' and mask_path is None: mask_path = os.path.join(d,f)
    return img_path, mask_path

# ===========================================================================
# ЧАСТЬ 3: СТАТИСТИЧЕСКИЙ АНАЛИЗ (evaluate_features.py)
# ===========================================================================

def calculate_class_stats(dataset, mask, class_id):
    """Вектор средних + ковариационная матрица для пикселей класса."""
    flat = mask.flatten()
    X    = dataset.reshape(-1, dataset.shape[-1])[flat == class_id]
    if len(X) < 5:
        print(f"   Класс {class_id}: недостаточно пикселей ({len(X)})")
        return None, None
    print(f"   Класс {class_id} ({CLASS_NAMES.get(class_id,'?')}): {len(X)} пкс")
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0: cov = np.array([[float(cov)]])
    cov += np.eye(cov.shape[0]) * 1e-6
    return np.mean(X, axis=0), cov


def mahalanobis_distance(m1, m2, cov1, cov2):
    """D_M = sqrt((m1-m2)^T · ((Σ1+Σ2)/2)^{-1} · (m1-m2))"""
    try:
        inv = np.linalg.inv((cov1+cov2)/2)
        d   = m1-m2
        return float(np.sqrt(d @ inv @ d))
    except np.linalg.LinAlgError:
        return np.nan


def bhattacharyya_distance(m1, m2, cov1, cov2):
    """
    D_B = (1/8)(m1-m2)^T Σ^{-1}(m1-m2) + (1/2)ln(det Σ / sqrt(det Σ1 · det Σ2))
    где Σ = (Σ1+Σ2)/2
    """
    cov  = (cov1+cov2)/2
    diff = m1-m2
    try:
        term1 = 0.125 * float(diff @ np.linalg.inv(cov) @ diff)
        s1,ld1 = np.linalg.slogdet(cov)
        s2,ld2 = np.linalg.slogdet(cov1)
        s3,ld3 = np.linalg.slogdet(cov2)
        if s1<=0 or s2<=0 or s3<=0: return np.nan
        return term1 + 0.5*(ld1 - 0.5*(ld2+ld3))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_all_pairwise_distances(stats, classes):
    """Матрицы попарных расстояний Махаланобиса и Бхаттачарьи."""
    n      = len(classes)
    mm, mb = np.zeros((n,n)), np.zeros((n,n))
    lbl    = [f"C{c}" for c in classes]
    for i,c1 in enumerate(classes):
        for j,c2 in enumerate(classes):
            if i==j or c1 not in stats or c2 not in stats: continue
            m1,v1 = stats[c1]['mean'], stats[c1]['cov']
            m2,v2 = stats[c2]['mean'], stats[c2]['cov']
            mm[i,j] = mahalanobis_distance(m1,m2,v1,v2)
            mb[i,j] = bhattacharyya_distance(m1,m2,v1,v2)
    return (pd.DataFrame(mm, index=lbl, columns=lbl),
            pd.DataFrame(mb, index=lbl, columns=lbl))

# ===========================================================================
# ЧАСТЬ 4: FORWARD SELECTION — БХАТТАЧАРЬЯ (forward_selection_stats.py)
# ===========================================================================

def _bhatta_samples(X1, X2):
    """Расстояние Бхаттачарьи по двум выборкам (надёжная версия)."""
    if len(X1)<5 or len(X2)<5: return 0.0
    if X1.ndim==1: X1=X1.reshape(-1,1)
    if X2.ndim==1: X2=X2.reshape(-1,1)
    m1,m2 = np.mean(X1,axis=0), np.mean(X2,axis=0)
    c1,c2 = np.cov(X1,rowvar=False), np.cov(X2,rowvar=False)
    if c1.ndim==0: c1=np.array([[float(c1)]])
    if c2.ndim==0: c2=np.array([[float(c2)]])
    reg=np.eye(c1.shape[0])*1e-6; c1+=reg; c2+=reg
    cov=(c1+c2)/2
    try:
        term1=0.125*float((m1-m2)@np.linalg.inv(cov)@(m1-m2))
        s1,ld1=np.linalg.slogdet(cov)
        s2,ld2=np.linalg.slogdet(c1)
        s3,ld3=np.linalg.slogdet(c2)
        if s1<=0 or s2<=0 or s3<=0: return 0.0
        return float(term1+0.5*(ld1-0.5*(ld2+ld3)))
    except np.linalg.LinAlgError:
        return 0.0


def forward_selection_bhatta(dataset, mask, target_classes=(2,11),
                              eps=0.001, max_features=10):
    """
    Жадный (greedy) Forward Selection по критерию расстояния Бхаттачарьи.
    На каждом шаге добавляется признак с максимальным приростом D_B.
    Остановка: прирост < eps или достигнут лимит max_features.
    """
    h,w,c   = dataset.shape
    flat    = mask.flatten()
    X       = dataset.reshape(-1,c)
    X1,X2   = X[flat==target_classes[0]], X[flat==target_classes[1]]

    if len(X1)<10 or len(X2)<10:
        print(f"❌ Мало пикселей: C{target_classes[0]}={len(X1)}, C{target_classes[1]}={len(X2)}")
        return [],[]

    selected, cur, history = [], 0.0, []
    print(f"\n🔍 Forward Selection (Бхаттачарья): классы {target_classes}")

    for step in range(max_features):
        best_f, best_g = -1, -1.0
        for i in range(c):
            if i in selected: continue
            gain = _bhatta_samples(X1[:,selected+[i]], X2[:,selected+[i]]) - cur
            if gain > best_g: best_g,best_f = gain,i
        if best_g < eps or best_f==-1:
            print(f"  ⏹  Остановка на шаге {step}. Прирост {best_g:.5f} < {eps}")
            break
        selected.append(best_f); cur+=best_g; history.append(cur)
        print(f"  Шаг {step+1}: признак #{best_f:>2d}, D_B={cur:.4f} (+{best_g:.4f})")

    return selected, history

# ===========================================================================
# ЧАСТЬ 5: FORWARD SELECTION — ML / kNN  (forward_selection_ml.py)
# ===========================================================================

def forward_selection_ml(dataset, mask, target_classes=None,
                          eps=0.001, max_features=10, max_samples=20_000):
    """
    Forward Selection по точности kNN (k=5, 5-fold CV, macro accuracy).
    StandardScaler применяется перед kNN (евклидово расстояние чувствительно к шкале).
    Выборка ограничена max_samples для скорости кросс-валидации.
    """
    h,w,c = dataset.shape
    flat  = mask.flatten()
    X_all = dataset.reshape(-1,c)[flat>0]
    y_all = flat[flat>0]

    if target_classes is not None:
        sel   = np.isin(y_all, target_classes)
        X_all,y_all = X_all[sel],y_all[sel]

    if len(X_all)>max_samples:
        rng=np.random.default_rng(42)
        idx=rng.choice(len(X_all),max_samples,replace=False)
        X_all,y_all=X_all[idx],y_all[idx]

    print(f"\n🤖 Forward Selection (kNN): {len(X_all)} пкс, {len(np.unique(y_all))} классов")
    Xs       = StandardScaler().fit_transform(X_all)
    selected, cur, history = [], 0.0, []

    for step in range(max_features):
        best_f,best_a,best_g = -1,-1.0,-1.0
        for i in range(c):
            if i in selected: continue
            acc  = cross_val_score(KNeighborsClassifier(5,n_jobs=-1),
                                   Xs[:,selected+[i]], y_all,
                                   cv=5, scoring='accuracy').mean()
            gain = acc-cur
            if gain>best_g: best_g,best_f,best_a=gain,i,acc
        if best_g<eps or best_f==-1:
            print(f"  ⏹  Остановка на шаге {step}. Прирост {best_g:.4f} < {eps}")
            break
        selected.append(best_f); cur=best_a; history.append(cur)
        print(f"  Шаг {step+1}: признак #{best_f:>2d}, Acc={cur:.4f} (+{best_g:.4f})")

    return selected, history

# ===========================================================================
# ЧАСТЬ 6: ДЕМО-ДАННЫЕ
# ===========================================================================

def _generate_synthetic_data(H=256, W=256, seed=42):
    """
    Синтетические данные для тестирования без реальных снимков.
    Два основных класса (2 — город, 11 — лес) + два дополнительных.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 220, size=(H,W), dtype=np.uint8)
    mask= np.zeros((H,W), dtype=np.uint8)

    # Квадранты
    mask[:H//2, :W//2] = 2    # город (яркий)
    mask[H//2:,  W//2:]= 11   # лес   (тёмный, шумный)
    mask[:H//4,  W//2:]= 3    # промзона
    mask[H//2:, :W//4] = 14   # луга

    img[:H//2,:W//2] = np.clip(rng.normal(160,20,(H//2,W//2)), 80,255).astype(np.uint8)
    img[H//2:, W//2:]= np.clip(rng.normal(90, 35,(H//2,W//2)), 20,200).astype(np.uint8)
    return img, mask

# ===========================================================================
# ЧАСТЬ 7: ВИЗУАЛИЗАЦИЯ — 10 РИСУНКОВ
# ===========================================================================

# --------------------------------------------------------------------------- [1]
def plot_feature_correlation(dataset, mask, names, out_dir):
    """
    Рисунок 1. Матрица корреляций Пирсона признакового пространства.

    Показывает линейную зависимость между признаками на всей выборке
    помеченных пикселей. Пары с |r| > 0.90 считаются избыточными
    и могут быть исключены корреляционным фильтром перед Forward Selection.
    """
    X = dataset.reshape(-1, dataset.shape[-1])[mask.flatten() > 0]
    if len(X) < 10:
        print("  ⚠️  Мало данных — рисунок 1 пропущен"); return

    corr = np.corrcoef(X.T)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Коэффициент корреляции Пирсона r')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)

    # Пороговые линии |r| = 0.90
    thresh = 0.90
    mask_high = np.abs(corr) > thresh
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                val = corr[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.5, color=color,
                        fontweight='bold' if abs(val) > thresh else 'normal')

    ax.set_title(
        'Рисунок 1. Матрица корреляций Пирсона признакового пространства\n'
        f'(выделены пары с |r| > {thresh} — кандидаты на фильтрацию)',
        fontsize=12, fontweight='bold', pad=12)

    # Рамки вокруг высоко-коррелированных ячеек
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and mask_high[i, j]:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor='gold', linewidth=1.5))

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_01_feature_correlation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 1: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [2]
def plot_class_distributions(dataset, mask, names, out_dir,
                              focus=('Mean', 'Std', 'Rho_Avg')):
    """
    Рисунок 2. KDE-гистограммы распределений ключевых признаков по классам.

    Чем меньше перекрытие распределений двух классов по признаку,
    тем выше его информативность (тем больше расстояние Бхаттачарьи).
    """
    from scipy.ndimage import gaussian_filter1d
    classes   = sorted(c for c in np.unique(mask) if c > 0)[:6]
    flat      = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])
    avail     = [f for f in focus if f in names] or names[:3]

    fig, axes = plt.subplots(1, len(avail), figsize=(5*len(avail), 5))
    if len(avail) == 1: axes = [axes]

    for ax, fname in zip(axes, avail):
        idx = names.index(fname)
        for ci, cls in enumerate(classes):
            px = data_flat[flat == cls, idx]
            if len(px) < 5: continue
            col = PALETTE.get(cls, DEFAULT_COLORS[ci % 10])
            cnts, edges = np.histogram(px, bins=60, density=True)
            ctrs = (edges[:-1]+edges[1:])/2
            ax.fill_between(ctrs, gaussian_filter1d(cnts, 2),
                            alpha=0.30, color=col)
            ax.plot(ctrs, gaussian_filter1d(cnts, 2),
                    color=col, lw=2, label=_label(cls))
        ax.set_title(f'Признак: {fname}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Значение'); ax.set_ylabel('Плотность вероятности')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Рисунок 2. Распределения ключевых признаков по классам\n'
        '(сглаженные KDE-гистограммы)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_02_class_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 2: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [3]
def plot_bhatta_heatmap(df_bhatta, out_dir):
    """
    Рисунок 3. Тепловая карта попарных расстояний Бхаттачарьи.

    Более высокое значение D_B — лучшая разделимость пары классов.
    Учитывает различие как в центрах, так и в форме ковариационных эллипсоидов.
    """
    arr  = df_bhatta.values.copy().astype(float)
    np.fill_diagonal(arr, np.nan)
    data = pd.DataFrame(arr, index=df_bhatta.index, columns=df_bhatta.columns)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ccc',
                cbar_kws={'label': 'Расстояние Бхаттачарьи D_B'})
    ax.set_title(
        'Рисунок 3. Попарные расстояния Бхаттачарьи между классами\n'
        '(полный признаковый набор, диагональ = NaN)',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_03_bhatta_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 3: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [4]
def plot_maha_heatmap(df_maha, out_dir):
    """
    Рисунок 4. Тепловая карта попарных расстояний Махаланобиса.

    В отличие от D_B учитывает только смещение центров классов,
    нормируя на усреднённую ковариацию. Сравнение рисунков 3 и 4
    показывает, в каких парах форма распределений вносит вклад в разделимость.
    """
    arr  = df_maha.values.copy().astype(float)
    np.fill_diagonal(arr, np.nan)
    data = pd.DataFrame(arr, index=df_maha.index, columns=df_maha.columns)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(data, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ccc',
                cbar_kws={'label': 'Расстояние Махаланобиса D_M'})
    ax.set_title(
        'Рисунок 4. Попарные расстояния Махаланобиса между классами\n'
        '(полный признаковый набор, диагональ = NaN)',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_04_maha_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 4: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [5]
def plot_bhatta_forward_selection(history, sel_names, out_dir):
    """
    Рисунок 5. Кривая Forward Selection по критерию Бхаттачарьи.

    Левая ось — накопленное D_B, правая — прирост на каждом шаге.
    «Колено» кривой указывает на оптимальный размер подмножества признаков.
    """
    if not history:
        print("  ⚠️  Нет данных — рисунок 5 пропущен"); return

    x = list(range(1, len(history)+1))
    gains = [history[0]] + [history[i]-history[i-1] for i in range(1, len(history))]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(x, history, marker='o', lw=2.5, ms=8, color='#E63946',
             markerfacecolor='white', markeredgewidth=2.5, label='D_B (накопл.)', zorder=3)
    ax2.bar(x, gains, alpha=0.22, color='#E63946', label='Прирост D_B')

    for i, (xi, yi, nm) in enumerate(zip(x, history, sel_names)):
        ax1.annotate(nm, xy=(xi, yi),
                     xytext=(xi, yi + max(history)*0.05),
                     ha='center', fontsize=8, rotation=25, color='#333',
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.8))

    ax1.set_xlabel('Количество признаков (шаг отбора)', fontsize=11)
    ax1.set_ylabel('Накопленное расстояние Бхаттачарьи D_B', fontsize=11)
    ax2.set_ylabel('Прирост D_B на шаге', color='#E63946', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#E63946')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        'Рисунок 5. Forward Selection по критерию Бхаттачарьи\n'
        f'(пара классов {sel_names[0] if sel_names else "?"} — жадный алгоритм)',
        fontsize=12, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_05_forward_bhatta.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 5: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [6]
def plot_ml_forward_selection(history, sel_names, out_dir):
    """
    Рисунок 6. Кривая Forward Selection по точности kNN (5-fold CV).

    Сравнение с рисунком 5 позволяет оценить согласованность
    статистического и ML-критериев отбора признаков.
    """
    if not history:
        print("  ⚠️  Нет данных — рисунок 6 пропущен"); return

    x      = list(range(1, len(history)+1))
    pct    = [v*100 for v in history]
    gains  = [pct[0]] + [pct[i]-pct[i-1] for i in range(1, len(pct))]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(x, pct, marker='s', lw=2.5, ms=8, color='#2A9D8F',
             markerfacecolor='white', markeredgewidth=2.5, label='Accuracy (CV)', zorder=3)
    ax2.bar(x, gains, alpha=0.22, color='#2A9D8F', label='Прирост Acc')

    for xi, yi, nm in zip(x, pct, sel_names):
        ax1.annotate(nm, xy=(xi, yi),
                     xytext=(xi, yi + max(pct)*0.04),
                     ha='center', fontsize=8, rotation=25, color='#333',
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.8))

    # Зоны качества
    ax1.axhspan(90, 105, alpha=0.06, color='green')
    ax1.axhspan(70, 90,  alpha=0.06, color='yellow')
    ax1.axhspan(0,  70,  alpha=0.06, color='red')
    ax1.text(len(x)*0.98, 97,  '>90% — отлично',  ha='right', fontsize=8, color='darkgreen')
    ax1.text(len(x)*0.98, 80,  '70–90% — хорошо', ha='right', fontsize=8, color='olive')
    ax1.text(len(x)*0.98, 55,  '<70% — слабо',    ha='right', fontsize=8, color='red')

    ax1.set_ylim(0, 108)
    ax1.set_xlabel('Количество признаков (шаг отбора)', fontsize=11)
    ax1.set_ylabel('Точность классификации, %', fontsize=11)
    ax2.set_ylabel('Прирост точности, п.п.', color='#2A9D8F', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='#2A9D8F')
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(
        'Рисунок 6. Forward Selection по точности kNN (5-fold CV, k=5)\n'
        '(все классы маски, macro accuracy)',
        fontsize=12, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_06_forward_ml.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 6: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [7]
def plot_kde_ellipsoids(dataset, mask, names, out_dir, cls_pair=(2,11)):
    """
    Рисунок 7. KDE-эллипсоиды рассеяния двух классов (Mean vs Rho_Avg).

    Форма контуров плотности отражает геометрию ковариационных эллипсоидов,
    используемых в расстоянии Бхаттачарьи. Звёздочки — центроиды.
    """
    from scipy.stats import gaussian_kde
    flat      = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])
    fx = 'Mean'    if 'Mean'    in names else names[0]
    fy = 'Rho_Avg' if 'Rho_Avg' in names else (names[1] if len(names)>1 else names[0])
    ix, iy = names.index(fx), names.index(fy)

    fig, ax = plt.subplots(figsize=(8, 7))
    for cls in cls_pair:
        px = data_flat[flat == cls]
        if len(px) < 5: continue
        col = PALETTE.get(cls, DEFAULT_COLORS[0])
        xv, yv = px[:, ix], px[:, iy]
        try:
            kde = gaussian_kde(np.vstack([xv, yv]), bw_method=0.3)
            xg = np.linspace(xv.min(), xv.max(), 100)
            yg = np.linspace(yv.min(), yv.max(), 100)
            XX, YY = np.meshgrid(xg, yg)
            Z = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contourf(XX, YY, Z, levels=6, alpha=0.25, colors=[col]*6)
            ax.contour( XX, YY, Z, levels=6, colors=[col], linewidths=1.5)
        except Exception:
            ax.scatter(xv[::20], yv[::20], s=10, alpha=0.3, color=col)
        ax.scatter(np.mean(xv), np.mean(yv), marker='*', s=250,
                   color=col, edgecolors='black', zorder=5, label=_label(cls))

    ax.set_xlabel(f'Признак: {fx}', fontsize=11)
    ax.set_ylabel(f'Признак: {fy}', fontsize=11)
    ax.set_title(
        f'Рисунок 7. KDE-эллипсоиды рассеяния классов\n'
        f'(проекция: {fx} vs {fy}, ★ — центроид)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_07_kde_ellipsoids.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 7: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [8]
def plot_feature_histograms(dataset, mask, names, out_dir, cls_pair=(2,11)):
    """
    Рисунок 8. Гистограммы всех признаков для пары классов.

    Площадь пересечения гистограмм ≈ вероятность ошибки классификации
    по одному признаку (байесовская граница Байеса).
    """
    flat      = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])
    c         = len(names)
    cols      = 4
    rows      = (c + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3.2))
    axes = axes.flatten()

    for i, fname in enumerate(names):
        ax = axes[i]
        for cls in cls_pair:
            px  = data_flat[flat == cls, i]
            if len(px) < 5: continue
            col = PALETTE.get(cls, DEFAULT_COLORS[0])
            ax.hist(px, bins=45, density=True, alpha=0.50,
                    color=col, label=f'C{cls}', edgecolor='none')
        ax.set_title(fname, fontsize=9, fontweight='bold')
        ax.set_xlabel('Значение', fontsize=7)
        ax.set_ylabel('Плотность', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'Рисунок 8. Гистограммы признаков для классов C{cls_pair[0]} и C{cls_pair[1]}\n'
        f'({CLASS_NAMES.get(cls_pair[0],"?")} | {CLASS_NAMES.get(cls_pair[1],"?")})',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_08_feature_histograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 8: {os.path.basename(path)}")


# --------------------------------------------------------------------------- [9]
def plot_boxplot_by_class(dataset, mask, names, out_dir, max_cls=6):
    """
    Рисунок 9. Box-plot нормализованных признаков по всем классам.

    Признаки нормированы в [0,1] для сопоставимости. Показывает медиану,
    IQR и выбросы — полезно для оценки межклассовых различий одновременно
    по всем признакам.
    """
    flat      = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1]).astype(np.float64)
    classes   = sorted(c for c in np.unique(mask) if c > 0)[:max_cls]

    # Нормализация [0,1]
    for i in range(data_flat.shape[1]):
        mn,mx = data_flat[:,i].min(), data_flat[:,i].max()
        if mx>mn: data_flat[:,i] = (data_flat[:,i]-mn)/(mx-mn)

    n_feat = len(names)
    cols   = 2
    rows   = (n_feat + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*3.5))
    axes = axes.flatten()

    for i, fname in enumerate(names):
        ax = axes[i]
        groups, lbls, cols_bp = [], [], []
        for cls in classes:
            px = data_flat[flat == cls, i]
            if len(px)<5: continue
            groups.append(px); lbls.append(f'C{cls}')
            cols_bp.append(PALETTE.get(cls, DEFAULT_COLORS[classes.index(cls)%10]))

        bp = ax.boxplot(groups, patch_artist=True, notch=False, widths=0.5,
                        medianprops=dict(color='black', lw=2))
        for patch, col in zip(bp['boxes'], cols_bp):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(lbls)+1)); ax.set_xticklabels(lbls, fontsize=8)
        ax.set_title(fname, fontsize=9, fontweight='bold')
        ax.set_ylabel('Норм. значение [0,1]', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'Рисунок 9. Box-plot нормализованных признаков по классам\n'
        '(медиана, IQR, выбросы — все классы маски)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_09_boxplot_classes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  ✅ Рисунок 9: {os.path.basename(path)}")


# ===========================================================================
# ЧАСТЬ 8: ГЛАВНЫЙ ПАЙПЛАЙН
# ===========================================================================

def main():
    """
    Полный Multi-Patch пайплайн курсовой работы (MultiSenGE):
      1. Загрузка нескольких случайных патчей (или демо-данных)
      2. Вычисление признакового пространства для каждого патча
      3. Объединение пикселей всех патчей в единую статистическую выборку
      4. Субдискретизация до MAX_PIXELS_TOTAL
      5. Статистический анализ (Махаланобис + Бхаттачарья)
      6. Forward Selection — Бхаттачарья (пара классов 2↔11)
      7. Forward Selection — kNN 5-fold CV (все классы)
      8. Построение 9 рисунков (graph_01 – graph_09)
      9. Итоговый отчёт в консоль
    """
    print("=" * 70)
    print("  Курсовая работа: Статистические методы отбора признаков")
    print("  при классификации космических изображений (MultiSenGE)")
    print(f"  Режим: Multi-Patch ({N_PATCHES} патчей, seed={RANDOM_SEED})")
    print("=" * 70)

    # Директории
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir      = os.path.join(project_root, 'output')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📁 Графики → {out_dir}")

    # -------------------------------------------------------------------
    # ШАГ 1: Загрузка / генерация данных
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 1: Загрузка данных")
    print("─" * 60)

    X_global = None
    y_global = None
    names    = None

    use_demo  = True

    # --- Попытка загрузить реальные патчи ---
    if HAS_RASTERIO:
        data_dir = os.path.join(project_root, 'data')
        lists_dir = os.path.join(project_root, 'lists')

        # Проверяем наличие файлов-списков
        has_lists = (
            os.path.isfile(os.path.join(lists_dir, 'out_s2_pref.txt')) and
            os.path.isfile(os.path.join(lists_dir, 'out_gr_pref.txt'))
        )

        if has_lists:
            try:
                pairs = select_random_pairs(n_patches=N_PATCHES, seed=RANDOM_SEED)

                for patch_idx, (s2_name, gr_name) in enumerate(pairs, 1):
                    print(f"\n  Патч {patch_idx}/{len(pairs)}: {s2_name}")
                    try:
                        raw_img, patch_mask = load_pair(s2_name, gr_name, data_dir)

                        # Нормализация
                        if raw_img.ndim == 3 and USE_MULTISPECTRAL:
                            # Мультиспектральный режим — передаём (C,H,W) → (H,W,C)
                            img_input = np.moveaxis(raw_img, 0, -1)  # (H,W,C)
                            # Нормируем каждый канал
                            for ch in range(img_input.shape[-1]):
                                mn, mx = img_input[:,:,ch].min(), img_input[:,:,ch].max()
                                img_input[:,:,ch] = (img_input[:,:,ch] - mn) / (mx - mn + 1e-8) * 255
                            img_input = img_input.astype(np.float32)
                            feat_dict = extract_all_features(img_input, window_size=WINDOW_SIZE,
                                                             is_multispectral=True)
                        else:
                            gray = np.mean(raw_img, axis=0) if raw_img.ndim == 3 else raw_img.squeeze()
                            mn, mx = gray.min(), gray.max()
                            img_norm = ((gray - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
                            feat_dict = extract_all_features(img_norm, window_size=WINDOW_SIZE,
                                                             is_multispectral=False)

                        cube, patch_names = make_feature_sandwich(feat_dict)

                        # Выравниваем маску по размеру куба, если нужно
                        if patch_mask.shape != cube.shape[:2]:
                            patch_mask = zoom(
                                patch_mask,
                                (cube.shape[0] / patch_mask.shape[0],
                                 cube.shape[1] / patch_mask.shape[1]),
                                order=0
                            ).astype(np.uint8)

                        X_patch, y_patch = build_global_dataset(cube, patch_mask)

                        if names is None:
                            names = patch_names

                        if X_global is None:
                            X_global = X_patch
                            y_global = y_patch
                        else:
                            X_global = np.vstack([X_global, X_patch])
                            y_global = np.concatenate([y_global, y_patch])

                        print(f"    ✅ +{len(X_patch):,} пкс | итого: {len(X_global):,}")

                    except Exception as e:
                        print(f"    ⚠️  Ошибка патча {s2_name}: {e}")

                if X_global is not None and len(X_global) > 100:
                    use_demo = False
                    print(f"\n  ✅ Загружено {len(pairs)} патчей: {len(X_global):,} пикселей")

            except Exception as e:
                print(f"  ⚠️  Ошибка загрузки патчей: {e}. Переключаюсь на демо-данные.")

        else:
            # Пробуем найти хотя бы один .tif
            img_path, mask_path = auto_find_data(project_root)
            if img_path and mask_path:
                print(f"  Одиночный снимок:  {os.path.basename(img_path)}")
                try:
                    with rasterio.open(mask_path) as s: patch_mask = s.read(1).astype(np.uint8)
                    with rasterio.open(img_path)  as s: raw = s.read().astype(np.float32)
                    gray = np.mean(raw, axis=0) if raw.ndim == 3 else raw.squeeze()
                    mn, mx = gray.min(), gray.max()
                    img_norm = ((gray - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
                    if patch_mask.shape != img_norm.shape:
                        patch_mask = zoom(
                            patch_mask,
                            (img_norm.shape[0] / patch_mask.shape[0],
                             img_norm.shape[1] / patch_mask.shape[1]),
                            order=0
                        ).astype(np.uint8)
                    feat_dict = extract_all_features(img_norm, window_size=WINDOW_SIZE,
                                                     is_multispectral=False)
                    cube, names = make_feature_sandwich(feat_dict)
                    X_global, y_global = build_global_dataset(cube, patch_mask)
                    use_demo = False
                    print(f"  ✅ Одиночный патч: {len(X_global):,} пикселей")
                except Exception as e:
                    print(f"  ❌ Ошибка: {e}. Переключаюсь на демо-данные.")

    # --- Демо-режим ---
    if use_demo:
        print("  🎲 Генерация синтетических данных (Multi-Patch демо, 4×256×256)...")
        X_parts, y_parts = [], []
        for seed_offset in range(N_PATCHES):
            img_norm, patch_mask = _generate_synthetic_data(
                H=256, W=256, seed=RANDOM_SEED + seed_offset
            )
            feat_dict = extract_all_features(img_norm, window_size=WINDOW_SIZE,
                                             is_multispectral=False)
            cube, names = make_feature_sandwich(feat_dict)
            X_p, y_p = build_global_dataset(cube, patch_mask)
            X_parts.append(X_p)
            y_parts.append(y_p)

        X_global = np.vstack(X_parts)
        y_global = np.concatenate(y_parts)
        print(f"  ✅ Синтетика: {N_PATCHES} патчей, {len(X_global):,} пикселей")

    # -------------------------------------------------------------------
    # ШАГ 2: Субдискретизация
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print(f"ШАГ 2: Субдискретизация (лимит {MAX_PIXELS_TOTAL:,} пкс)")
    print("─" * 60)
    X_global, y_global = subsample_dataset(X_global, y_global,
                                            max_samples=MAX_PIXELS_TOTAL,
                                            seed=RANDOM_SEED)
    print(f"  📊 Итоговая выборка: {len(X_global):,} пикселей, {len(names)} признаков")

    unique_cls = np.unique(y_global)
    unique_cls = unique_cls[unique_cls > 0]
    print(f"  Классы в выборке: {unique_cls.tolist()}")

    # -------------------------------------------------------------------
    # Восстанавливаем псевдо-куб для совместимости с функциями графиков
    # -------------------------------------------------------------------
    dataset, mask = rebuild_feature_cube(X_global, y_global)

    # -------------------------------------------------------------------
    # ШАГ 3: Статистический анализ
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 3: Расстояния Махаланобиса и Бхаттачарьи")
    print("─" * 60)
    stats = {}
    for cls in unique_cls:
        mv, cm = calculate_class_stats(dataset, mask, int(cls))
        if mv is not None:
            stats[int(cls)] = {'mean': mv, 'cov': cm}

    classes_list = sorted(stats.keys())
    df_maha, df_bhatt = compute_all_pairwise_distances(stats, classes_list)
    print("\n  Расстояния Бхаттачарьи:"); print(df_bhatt.round(3).to_string())
    print("\n  Расстояния Махаланобиса:"); print(df_maha.round(3).to_string())

    # -------------------------------------------------------------------
    # ШАГ 4: Forward Selection — Бхаттачарья
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 4: Forward Selection — Бхаттачарья")
    print("─" * 60)
    pair = None
    if 2 in unique_cls and 11 in unique_cls:
        pair = (2, 11)
    elif len(unique_cls) >= 2:
        pair = (int(unique_cls[0]), int(unique_cls[1]))

    sel_b, hist_b = [], []
    if pair:
        sel_b, hist_b = forward_selection_bhatta(
            dataset, mask, pair, eps=0.001, max_features=10
        )
        print(f"\n  🏆 Бхаттачарья: {[names[i] for i in sel_b]}")

    # -------------------------------------------------------------------
    # ШАГ 5: Forward Selection — kNN
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 5: Forward Selection — kNN (5-fold CV)")
    print("─" * 60)
    sel_m, hist_m = forward_selection_ml(
        dataset, mask,
        target_classes=[int(c) for c in unique_cls],
        eps=0.001, max_features=10
    )
    print(f"\n  🏆 kNN: {[names[i] for i in sel_m]}")

    # -------------------------------------------------------------------
    # ШАГ 6: Построение рисунков
    # -------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 6: Построение рисунков (1–10)")
    print("─" * 60)

    clsp = pair if pair else (int(unique_cls[0]), int(unique_cls[0]))

    plot_feature_correlation(dataset, mask, names, out_dir)
    plot_class_distributions(dataset, mask, names, out_dir)
    plot_bhatta_heatmap(df_bhatt, out_dir)
    plot_maha_heatmap(df_maha, out_dir)
    plot_bhatta_forward_selection(hist_b, [names[i] for i in sel_b], out_dir)
    plot_ml_forward_selection(hist_m, [names[i] for i in sel_m], out_dir)
    plot_kde_ellipsoids(dataset, mask, names, out_dir, cls_pair=clsp)
    plot_feature_histograms(dataset, mask, names, out_dir, cls_pair=clsp)
    plot_boxplot_by_class(dataset, mask, names, out_dir)
    # (graph_10 — сравнительная таблица строится отдельно в анализе результатов)

    # -------------------------------------------------------------------
    # ИТОГОВЫЙ ОТЧЁТ
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)
    print(f"\n  Количество патчей     : {N_PATCHES}")
    print(f"  Признаков всего       : {len(names)}")
    print(f"  Объём выборки         : {len(X_global):,}")
    print(f"  Классов в выборке     : {len(classes_list)}")

    if sel_b:
        print(f"\n  Forward Selection (Бхаттачарья), классы {pair}:")
        for r, i in enumerate(sel_b, 1):
            d = f'{hist_b[r-1]:.4f}' if r-1 < len(hist_b) else '—'
            print(f"    {r}. {names[i]:<18s}  D_B = {d}")

    if sel_m:
        print(f"\n  Forward Selection (kNN), все классы:")
        for r, i in enumerate(sel_m, 1):
            a = f'{hist_m[r-1]*100:.1f}%' if r-1 < len(hist_m) else '—'
            print(f"    {r}. {names[i]:<18s}  Acc = {a}")

    common = set(names[i] for i in sel_b) & set(names[i] for i in sel_m)
    if common:
        print(f"\n  ✨ Отобрано обоими методами: {sorted(common)}")

    print(f"\n  📂 Все рисунки: {out_dir}")
    print("\n  ✅ Эксперимент завершён")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()