"""
=============================================================================
combined_module.py
=============================================================================
Курсовая работа: «Статистические методы отбора признаков при классификации
космических изображений (на примере датасета MultiSenGE)»

Объединённый модуль включает:
  1. fast_features.py  — вычисление признакового пространства
  2. loader.py         — загрузка данных Sentinel-2
  3. evaluate_features.py — анализ классов: расстояния Махаланобиса и Бхаттачарьи
  4. forward_selection_stats.py — статистический Forward Selection (Бхаттачарья)
  5. forward_selection_ml.py   — ML Forward Selection (kNN + 5-fold CV)

Графики (минимум 8):
  [1]  Матрица корреляций признаков
  [2]  Распределение яркостей классов по признаку Original / Mean
  [3]  Тепловая карта попарных расстояний Бхаттачарьи
  [4]  Тепловая карта попарных расстояний Махаланобиса
  [5]  Кривая Forward Selection (статистический — Бхаттачарья)
  [6]  Кривая Forward Selection (ML — kNN Accuracy)
  [7]  Совместное распределение двух классов (KDE-эллипсоиды)
  [8]  Гистограммы ключевых признаков для двух классов
  [9]  Нормализованное признаковое пространство (boxplot по классам)
  [10] Карта отобранных признаков на снимке (псевдоцветная визуализация)
=============================================================================
"""

# ---------------------------------------------------------------------------
# Импорты
# ---------------------------------------------------------------------------
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # безголовый режим, убрать если нужен GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.ndimage import uniform_filter

# Опциональные зависимости (геопространственные данные)
try:
    import rasterio
    from scipy.ndimage import zoom
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("⚠️  rasterio не установлен — модуль запустится в демо-режиме с синтетическими данными.")

# Машинное обучение
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 1: ВЫЧИСЛЕНИЕ ПРИЗНАКОВ (fast_features.py)
# ============================================================================
# ---------------------------------------------------------------------------

def get_fast_stats(image: np.ndarray, window_size: int):
    """
    Вычисляет локальное среднее, дисперсию и СКО за O(1) на пиксель.

    Использует интегральные изображения через uniform_filter, что
    позволяет обрабатывать снимки любого размера без «скользящего окна».

    Parameters
    ----------
    image       : (H, W) float-массив
    window_size : размер квадратного окна (нечётный, например 15)

    Returns
    -------
    mean, var, std : каждый (H, W) float64
    """
    n = window_size
    img64 = image.astype(np.float64)
    mean   = uniform_filter(img64,    size=n, mode='mirror')
    sq_mean = uniform_filter(img64**2, size=n, mode='mirror')
    var = np.maximum(sq_mean - mean**2, 0.0)
    std = np.sqrt(var)
    return mean, var, std


def calc_directional_rho(image: np.ndarray, mean: np.ndarray,
                          var: np.ndarray, window_size: int) -> dict:
    """
    Направленные коэффициенты корреляции по 4 направлениям:
    0° (→), 90° (↓), 45° (↘), 135° (↙).

    Рассчитывается как нормированная локальная ковариация пикселя
    с его ближайшим соседом в каждом из направлений.

    Parameters
    ----------
    image, mean, var : (H, W) массивы
    window_size      : размер окна усреднения

    Returns
    -------
    rhos : dict с ключами '0', '90', '45', '135' → (H, W) в [-1, 1]
    """
    directions = {'0': (0, 1), '90': (1, 0), '45': (1, 1), '135': (1, -1)}
    rhos = {}
    eps = 1e-10
    for angle, (dy, dx) in directions.items():
        shifted  = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        f_adj    = uniform_filter(image * shifted, size=window_size, mode='mirror')
        rho      = (f_adj - mean**2) / (var + eps)
        rhos[angle] = np.clip(rho, -1.0, 1.0)
    return rhos


def compute_spectral_indices(img_10ch: np.ndarray) -> dict:
    """
    Вычисляет спектральные вегетационные и городские индексы
    для 10-канального снимка Sentinel-2.

    Порядок каналов: B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12.

    Возвращаемые индексы
    --------------------
    NDVI  : нормализованный вегетационный индекс  (B8-B4)/(B8+B4)
    NDWI  : нормализованный водный индекс         (B3-B8)/(B3+B8)
    NDBI  : нормализованный индекс застройки      (B11-B8)/(B11+B8)
    Total_Brightness : сумма всех 10 каналов
    Norm_Bx : нормированные (инварианты освещённости) значения каналов

    Parameters
    ----------
    img_10ch : (10, H, W) float-массив

    Returns
    -------
    indices : dict str → (H, W) float32
    """
    eps = 1e-8
    B2, B3, B4, B8 = img_10ch[0], img_10ch[1], img_10ch[2], img_10ch[3]
    B11, B12        = img_10ch[8], img_10ch[9]

    ndvi = (B8  - B4)  / (B8  + B4  + eps)
    ndwi = (B3  - B8)  / (B3  + B8  + eps)
    ndbi = (B11 - B8)  / (B11 + B8  + eps)
    total_brightness = np.sum(img_10ch, axis=0)
    norm_channels    = img_10ch / (total_brightness + eps)

    indices = {
        'NDVI': ndvi.astype(np.float32),
        'NDWI': ndwi.astype(np.float32),
        'NDBI': ndbi.astype(np.float32),
        'Total_Brightness': total_brightness.astype(np.float32),
    }
    for i, name in enumerate(['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']):
        indices[f'Norm_{name}'] = norm_channels[i].astype(np.float32)

    return indices


def extract_all_features(image: np.ndarray, window_size: int = 15,
                          is_multispectral: bool = False) -> dict:
    """
    Главная функция генерации признакового пространства.

    Формирует полный набор признаков:
      - спектральные индексы (NDVI, NDWI, NDBI и нормированные каналы)
        — только при is_multispectral=True
      - локальное среднее Mean и стандартное отклонение Std
      - 4 направленных корреляции (Rho_0, Rho_90, Rho_45, Rho_135)
      - агрегаты: Rho_Avg (среднее), Rho_Range (размах)
      - для одноканального входа: сохраняется Original

    Parameters
    ----------
    image           : (C, H, W) многоканальный или (H, W) одноканальный
    window_size     : размер скользящего окна (рекомендуется 15)
    is_multispectral: True — считать спектральные индексы

    Returns
    -------
    feature_space : dict str → (H, W) float32/float64
    """
    feature_space = {}

    # --- Спектральные признаки ---
    if is_multispectral and image.ndim == 3:
        print("  📡 Вычисление спектральных индексов (NDVI, NDWI, NDBI)...")
        indices = compute_spectral_indices(image)
        feature_space.update(indices)
        gray_image = np.mean(image, axis=0).astype(np.float32)
    else:
        gray_image = image.astype(np.float32)
        feature_space['Original'] = gray_image

    # --- Локальные статистики ---
    print(f"  📐 Локальные статистики (окно {window_size}×{window_size})...")
    mean, var, std = get_fast_stats(gray_image, window_size)
    feature_space['Mean'] = mean.astype(np.float32)
    feature_space['Std']  = std.astype(np.float32)

    # --- Текстурные корреляции ---
    print("  🧵 Направленные корреляции (0°, 90°, 45°, 135°)...")
    rhos = calc_directional_rho(gray_image, mean, var, window_size)
    feature_space['Rho_Avg']   = ((rhos['0'] + rhos['90'] + rhos['45'] + rhos['135']) / 4).astype(np.float32)
    feature_space['Rho_Range'] = (
        np.maximum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']]) -
        np.minimum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']])
    ).astype(np.float32)
    for angle in ('0', '90', '45', '135'):
        feature_space[f'Rho_{angle}'] = rhos[angle].astype(np.float32)

    return feature_space


def make_feature_sandwich(feature_dict: dict):
    """
    Собирает словарь признаков в трёхмерный массив (H, W, C).

    Parameters
    ----------
    feature_dict : dict str → (H, W)

    Returns
    -------
    dataset : (H, W, C) float32
    names   : list[str] — имена каналов в том же порядке
    """
    names    = list(feature_dict.keys())
    channels = [feature_dict[name].astype(np.float32) for name in names]
    dataset  = np.stack(channels, axis=-1)   # (H, W, C)
    return dataset, names


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 2: ЗАГРУЗКА ДАННЫХ (loader.py)
# ============================================================================
# ---------------------------------------------------------------------------

# Порядок каналов Sentinel-2 в датасете MultiSenGE
SENTINEL2_CHANNELS = {
    'B2': 1, 'B3': 2, 'B4': 3, 'B8': 4,
    'B5': 5, 'B6': 6, 'B7': 7, 'B8A': 8,
    'B11': 9, 'B12': 10
}

# Имена классов MultiSenGE (CORINE Land Cover)
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


def load_pair(s2_name: str, gr_name: str, data_dir: str = None):
    """
    Загружает пару файлов: 10-канальный снимок Sentinel-2 и эталонную маску.

    Parameters
    ----------
    s2_name  : имя файла снимка (.tif)
    gr_name  : имя файла маски (.tif)
    data_dir : корневая папка с данными (data/)

    Returns
    -------
    image : (10, H, W) float32  — значения отражательной способности
    mask  : (H, W)  uint8       — метки классов 1..14, 0 — фон
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio не установлен")
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

    s2_path = os.path.join(data_dir, 's2_pref', s2_name.strip())
    gr_path = os.path.join(data_dir, 'ground_reference', gr_name.strip())

    with rasterio.open(s2_path) as src:
        image = src.read().astype(np.float32)   # (10, H, W)
    with rasterio.open(gr_path) as src:
        mask = src.read(1).astype(np.uint8)     # (H, W)

    return image, mask


def get_file_lists(lists_dir: str = None):
    """
    Читает списки файлов из текстовых файлов out_s2_pref.txt / out_gr_pref.txt.

    Parameters
    ----------
    lists_dir : папка со списками (lists/)

    Returns
    -------
    s2_files, gr_files : list[str]
    """
    if lists_dir is None:
        lists_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lists')

    with open(os.path.join(lists_dir, 'out_s2_pref.txt'), 'r') as f:
        s2_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(lists_dir, 'out_gr_pref.txt'), 'r') as f:
        gr_files = [l.strip() for l in f if l.strip()]

    return s2_files, gr_files


def auto_find_data(project_root: str = None):
    """
    Автоматически ищет первую пару (снимок, маска) в структуре проекта.

    Returns
    -------
    img_path, mask_path : str | None
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir  = os.path.join(project_root, 'data')
    s2_dir    = os.path.join(data_dir, 's2_pref')
    mask_dir  = os.path.join(data_dir, 'ground_reference')

    img_path  = None
    mask_path = None

    for d, store in [(s2_dir, 'img'), (mask_dir, 'mask')]:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.lower().endswith(('.tif', '.tiff')):
                if store == 'img' and img_path is None:
                    img_path = os.path.join(d, fname)
                elif store == 'mask' and mask_path is None:
                    mask_path = os.path.join(d, fname)

    return img_path, mask_path


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 3: СТАТИСТИЧЕСКИЙ АНАЛИЗ КЛАССОВ (evaluate_features.py)
# ============================================================================
# ---------------------------------------------------------------------------

def calculate_class_stats(dataset: np.ndarray, mask: np.ndarray, class_id: int):
    """
    Извлекает пиксели заданного класса и вычисляет вектор средних и
    ковариационную матрицу.

    Parameters
    ----------
    dataset  : (H, W, C) float — признаковый куб
    mask     : (H, W) int     — маска классов
    class_id : int            — номер класса

    Returns
    -------
    mean_vec : (C,)  float64 | None
    cov_mat  : (C, C) float64 | None
    """
    mask_flat    = mask.flatten()
    data_flat    = dataset.reshape(-1, dataset.shape[-1])
    pixels       = data_flat[mask_flat == class_id]

    if len(pixels) < 5:
        print(f"   Класс {class_id}: недостаточно пикселей ({len(pixels)})")
        return None, None

    print(f"   Класс {class_id} ({CLASS_NAMES.get(class_id, '?')}): {len(pixels)} px")
    mean_vec = np.mean(pixels, axis=0)
    cov_mat  = np.cov(pixels, rowvar=False)
    if cov_mat.ndim == 0:
        cov_mat = np.array([[float(cov_mat)]])
    cov_mat += np.eye(cov_mat.shape[0]) * 1e-6   # регуляризация
    return mean_vec, cov_mat


def mahalanobis_distance(m1, m2, cov1, cov2) -> float:
    """
    Расстояние Махаланобиса при допущении общей (усреднённой) ковариации.

        D_M = sqrt((m1-m2)^T · Σ^{-1} · (m1-m2)),   Σ = (Σ1+Σ2)/2

    Высокое значение означает хорошую разделимость классов по положению
    центров в признаковом пространстве.
    """
    avg_cov = (cov1 + cov2) / 2
    try:
        inv_cov = np.linalg.inv(avg_cov)
        diff    = m1 - m2
        dist    = float(np.sqrt(diff @ inv_cov @ diff))
        return dist
    except np.linalg.LinAlgError:
        return np.nan


def bhattacharyya_distance(m1, m2, cov1, cov2) -> float:
    """
    Расстояние Бхаттачарьи.

    Учитывает как разницу центров, так и форму/объём эллипсоидов
    распределений, что делает его более полным критерием разделимости:

        D_B = (1/8)(m1-m2)^T Σ^{-1}(m1-m2)
              + (1/2) ln( det(Σ) / sqrt(det(Σ1)·det(Σ2)) )

    где Σ = (Σ1+Σ2)/2.
    """
    cov  = (cov1 + cov2) / 2
    diff = m1 - m2
    try:
        term1 = 0.125 * float(diff @ np.linalg.inv(cov) @ diff)
        sign1, ld_cov = np.linalg.slogdet(cov)
        sign2, ld_c1  = np.linalg.slogdet(cov1)
        sign3, ld_c2  = np.linalg.slogdet(cov2)
        if sign1 <= 0 or sign2 <= 0 or sign3 <= 0:
            return np.nan
        term2 = 0.5 * (ld_cov - 0.5 * (ld_c1 + ld_c2))
        return term1 + term2
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def compute_all_pairwise_distances(stats: dict, classes: list):
    """
    Вычисляет матрицы попарных расстояний Махаланобиса и Бхаттачарьи
    для всех пар классов.

    Parameters
    ----------
    stats   : {class_id: {'mean': ..., 'cov': ...}}
    classes : список идентификаторов классов

    Returns
    -------
    df_maha, df_bhatta : pd.DataFrame (симметричные матрицы)
    """
    n         = len(classes)
    mat_maha  = np.zeros((n, n))
    mat_bhatt = np.zeros((n, n))
    labels    = [f"C{c}" for c in classes]

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            if i == j:
                continue
            if c1 not in stats or c2 not in stats:
                continue
            m1, cov1 = stats[c1]['mean'], stats[c1]['cov']
            m2, cov2 = stats[c2]['mean'], stats[c2]['cov']
            mat_maha[i, j]  = mahalanobis_distance(m1, m2, cov1, cov2)
            mat_bhatt[i, j] = bhattacharyya_distance(m1, m2, cov1, cov2)

    df_maha  = pd.DataFrame(mat_maha,  index=labels, columns=labels)
    df_bhatt = pd.DataFrame(mat_bhatt, index=labels, columns=labels)
    return df_maha, df_bhatt


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 4: СТАТИСТИЧЕСКИЙ FORWARD SELECTION — БХАТТАЧАРЬЯ
# ============================================================================
# ---------------------------------------------------------------------------

def calculate_bhatta_dist(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Расстояние Бхаттачарьи для двух многомерных выборок.

    Надёжная реализация с регуляризацией и обработкой вырожденных случаев.
    Работает как для одного признака, так и для набора из N признаков.

    Parameters
    ----------
    X1, X2 : (N1, D) и (N2, D) — выборки пикселей двух классов
             (допустимо передать (N,) для одного признака)

    Returns
    -------
    float : расстояние Бхаттачарьи ≥ 0
    """
    if len(X1) < 5 or len(X2) < 5:
        return 0.0
    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    m1, m2 = np.mean(X1, axis=0), np.mean(X2, axis=0)
    c1 = np.cov(X1, rowvar=False)
    c2 = np.cov(X2, rowvar=False)
    if c1.ndim == 0: c1 = np.array([[float(c1)]])
    if c2.ndim == 0: c2 = np.array([[float(c2)]])

    reg = np.eye(c1.shape[0]) * 1e-6
    c1 += reg; c2 += reg
    cov = (c1 + c2) / 2

    try:
        inv_cov = np.linalg.inv(cov)
        diff    = m1 - m2
        term1   = 0.125 * float(diff @ inv_cov @ diff)
        s1, ld1 = np.linalg.slogdet(cov)
        s2, ld2 = np.linalg.slogdet(c1)
        s3, ld3 = np.linalg.slogdet(c2)
        if s1 <= 0 or s2 <= 0 or s3 <= 0:
            return 0.0
        term2 = 0.5 * (ld1 - 0.5 * (ld2 + ld3))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return 0.0


def forward_selection_bhatta(dataset: np.ndarray, mask: np.ndarray,
                               target_classes=(2, 11),
                               eps: float = 0.001,
                               max_features: int = 10):
    """
    Жадный (greedy) алгоритм отбора признаков — Forward Selection —
    по критерию расстояния Бхаттачарьи.

    На каждом шаге добавляется тот признак, который максимизирует прирост
    D_B(selected + {candidate}). Алгоритм останавливается, если прирост
    падает ниже порога eps или достигнут лимит max_features.

    Parameters
    ----------
    dataset        : (H, W, C) признаковый куб
    mask           : (H, W) маска классов
    target_classes : пара классов (class1, class2)
    eps            : порог остановки по приросту D_B
    max_features   : максимум отбираемых признаков

    Returns
    -------
    selected : list[int]   — индексы отобранных признаков
    history  : list[float] — накопленный D_B после каждого шага
    """
    h, w, c  = dataset.shape
    mask_flat = mask.flatten()
    feat_mat  = dataset.reshape(-1, c)

    X1 = feat_mat[mask_flat == target_classes[0]]
    X2 = feat_mat[mask_flat == target_classes[1]]

    if len(X1) < 10 or len(X2) < 10:
        print(f"❌ Недостаточно пикселей: класс {target_classes[0]}: {len(X1)}, "
              f"класс {target_classes[1]}: {len(X2)}")
        return [], []

    selected     = []
    current_dist = 0.0
    history      = []

    print(f"\n🔍 Forward Selection (Бхаттачарья): классы {target_classes}")
    for step in range(max_features):
        best_feat, best_gain = -1, -1.0
        for i in range(c):
            if i in selected:
                continue
            idxs = selected + [i]
            dist = calculate_bhatta_dist(X1[:, idxs], X2[:, idxs])
            gain = dist - current_dist
            if gain > best_gain:
                best_gain, best_feat = gain, i

        if best_gain < eps or best_feat == -1:
            print(f"  ⏹️  Остановка на шаге {step}. Прирост {best_gain:.5f} < {eps}")
            break

        selected.append(best_feat)
        current_dist += best_gain
        history.append(current_dist)
        print(f"  Шаг {step+1}: признак #{best_feat:>2d}, D_B = {current_dist:.4f} (+{best_gain:.4f})")

    return selected, history


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 5: ML FORWARD SELECTION — kNN + 5-FOLD CV
# ============================================================================
# ---------------------------------------------------------------------------

def forward_selection_ml(dataset: np.ndarray, mask: np.ndarray,
                          target_classes=None,
                          eps: float = 0.001,
                          max_features: int = 10,
                          max_samples: int = 20_000):
    """
    Жадный отбор признаков по приросту точности классификации kNN.

    Используется 5-кратная кросс-валидация (StratifiedKFold), метрика —
    macro accuracy. Масштабирование признаков (StandardScaler) применяется
    перед kNN, т.к. евклидово расстояние чувствительно к шкале.

    Parameters
    ----------
    dataset       : (H, W, C) признаковый куб
    mask          : (H, W) маска классов
    target_classes: список классов (None = все ненулевые)
    eps           : минимальный прирост точности для продолжения
    max_features  : лимит отбираемых признаков
    max_samples   : ограничение выборки (для скорости CV)

    Returns
    -------
    selected : list[int]   — индексы отобранных признаков
    history  : list[float] — точность CV после каждого шага
    """
    h, w, c   = dataset.shape
    mask_flat = mask.flatten()
    feat_mat  = dataset.reshape(-1, c)

    valid = mask_flat > 0
    X_all = feat_mat[valid]
    y_all = mask_flat[valid]

    if target_classes is not None:
        sel   = np.isin(y_all, target_classes)
        X_all = X_all[sel]; y_all = y_all[sel]

    if len(X_all) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_all), max_samples, replace=False)
        X_all = X_all[idx]; y_all = y_all[idx]

    print(f"\n🤖 Forward Selection (kNN): {len(X_all)} пикселей, "
          f"{len(np.unique(y_all))} классов")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    selected     = []
    current_acc  = 0.0
    history      = []

    for step in range(max_features):
        best_feat, best_acc, best_gain = -1, -1.0, -1.0
        for i in range(c):
            if i in selected:
                continue
            idxs    = selected + [i]
            X_sub   = X_scaled[:, idxs]
            clf     = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            scores  = cross_val_score(clf, X_sub, y_all, cv=5, scoring='accuracy')
            acc     = scores.mean()
            gain    = acc - current_acc
            if gain > best_gain:
                best_gain, best_feat, best_acc = gain, i, acc

        if best_gain < eps or best_feat == -1:
            print(f"  ⏹️  Остановка на шаге {step}. Прирост {best_gain:.4f} < {eps}")
            break

        selected.append(best_feat)
        current_acc = best_acc
        history.append(current_acc)
        print(f"  Шаг {step+1}: признак #{best_feat:>2d}, Accuracy = {current_acc:.4f} (+{best_gain:.4f})")

    return selected, history


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 6: ДЕМО-ДАННЫЕ (используются при отсутствии реальных снимков)
# ============================================================================
# ---------------------------------------------------------------------------

def _generate_synthetic_data(H=256, W=256, seed=42):
    """
    Генерирует синтетические данные, имитирующие признаковое пространство
    классификации земного покрова, для тестирования без реальных снимков.

    Создаёт два класса с разными статистическими характеристиками:
      - Класс 2  : прерывистая городская застройка (высокая яркость, низкая текстура)
      - Класс 11 : широколиственные леса           (средняя яркость, высокая текстура)
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(30, 220, size=(H, W), dtype=np.uint8)

    # Маска: верхний левый квадрант — класс 2, нижний правый — класс 11
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[:H//2, :W//2] = 2    # городская застройка
    mask[H//2:, W//2:] = 11   # лесной массив
    # Добавляем ещё несколько классов для полноты анализа
    mask[:H//4, W//2:] = 3
    mask[H//2:, :W//4] = 14

    # «Город» ярче, «лес» немного темнее со случайной текстурой
    img[:H//2, :W//2] = np.clip(
        rng.normal(160, 20, size=(H//2, W//2)), 80, 255).astype(np.uint8)
    img[H//2:, W//2:] = np.clip(
        rng.normal(90, 35, size=(H//2, W//2)), 20, 200).astype(np.uint8)

    return img, mask


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 7: ВИЗУАЛИЗАЦИЯ — 10 ГРАФИКОВ
# ============================================================================
# ---------------------------------------------------------------------------

PALETTE = {
    2:  '#E63946',   # городская застройка — красный
    3:  '#FF9F1C',   # промзона — оранжевый
    11: '#2A9D8F',   # леса — зелёно-синий
    14: '#8AC926',   # луга — салатовый
}
DEFAULT_COLORS = plt.cm.tab10.colors


def _safe_label(class_id: int) -> str:
    name = CLASS_NAMES.get(class_id, f'Класс {class_id}')
    if len(name) > 22:
        name = name[:20] + '…'
    return f"C{class_id}: {name}"


# --------------------------------------------------------------------------- GRAPH 1
def plot_feature_correlation(dataset: np.ndarray, mask: np.ndarray,
                              names: list, out_dir: str):
    """
    [График 1] Матрица корреляций Пирсона между признаками.

    Показывает, насколько линейно связаны признаки между собой.
    Высокая корреляция (|r|→1) означает избыточность; Forward Selection
    автоматически отсеивает такие признаки через критерий разделимости.
    """
    mask_flat = mask.flatten()
    valid     = mask_flat > 0
    X         = dataset.reshape(-1, dataset.shape[-1])[valid]

    if X.shape[0] < 10:
        print("  ⚠️  Недостаточно данных для матрицы корреляций")
        return

    corr = np.corrcoef(X.T)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Коэффициент корреляции Пирсона')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title('График 1. Матрица корреляций признакового пространства\n'
                 '(MultiSenGE, все классы)', fontsize=12, fontweight='bold')

    # Аннотации для небольших матриц
    if len(names) <= 12:
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(corr[i,j]) < 0.6 else 'white')

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_01_feature_correlation.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 1 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 2
def plot_class_distributions(dataset: np.ndarray, mask: np.ndarray,
                              names: list, out_dir: str,
                              focus_features=('Mean', 'Std', 'Rho_Avg')):
    """
    [График 2] Нормализованные гистограммы (KDE) ключевых признаков
    для основных классов.

    Позволяет визуально оценить разделимость классов по отдельным
    признакам. Чем меньше перекрытие распределений, тем информативнее признак.
    """
    classes   = sorted(c for c in np.unique(mask) if c > 0)[:6]
    mask_flat = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])

    available = [f for f in focus_features if f in names]
    if not available:
        available = names[:3]

    n_feat = len(available)
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]

    for ax, feat_name in zip(axes, available):
        idx = names.index(feat_name)
        for ci, cls in enumerate(classes):
            pixels = data_flat[mask_flat == cls, idx]
            if len(pixels) < 5:
                continue
            color = PALETTE.get(cls, DEFAULT_COLORS[ci % 10])
            # KDE через гистограмму scipy
            ax.hist(pixels, bins=50, density=True, alpha=0.45,
                    color=color, label=_safe_label(cls), edgecolor='none')
            # Smoothed line
            from scipy.ndimage import gaussian_filter1d
            counts, edges = np.histogram(pixels, bins=80, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            smooth  = gaussian_filter1d(counts, sigma=2)
            ax.plot(centers, smooth, color=color, linewidth=2)

        ax.set_title(f'Признак: {feat_name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Значение признака')
        ax.set_ylabel('Плотность вероятности')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('График 2. Распределения признаков по классам\n'
                 '(нормализованные гистограммы + сглаживание)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_02_class_distributions.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 2 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 3
def plot_bhatta_heatmap(df_bhatta: pd.DataFrame, out_dir: str):
    """
    [График 3] Тепловая карта попарных расстояний Бхаттачарьи.

    Более высокое значение → лучшая разделимость пары классов.
    Значения NaN обозначают вырожденную ковариацию (недостаточно пикселей).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    data = df_bhatta.copy()
    arr = data.values.copy(); np.fill_diagonal(arr, np.nan); data = pd.DataFrame(arr, index=data.index, columns=data.columns)

    sns.heatmap(data, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ddd',
                cbar_kws={'label': 'Расстояние Бхаттачарьи'})
    ax.set_title('График 3. Попарные расстояния Бхаттачарьи между классами\n'
                 '(полный признаковый набор)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_03_bhatta_heatmap.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 3 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 4
def plot_maha_heatmap(df_maha: pd.DataFrame, out_dir: str):
    """
    [График 4] Тепловая карта попарных расстояний Махаланобиса.

    В отличие от Бхаттачарьи учитывает только смещение центров классов,
    нормируя его по усреднённой ковариации.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    data = df_maha.copy()
    arr = data.values.copy(); np.fill_diagonal(arr, np.nan); data = pd.DataFrame(arr, index=data.index, columns=data.columns)

    sns.heatmap(data, ax=ax, cmap='Blues', annot=True, fmt='.2f',
                linewidths=0.5, linecolor='#ddd',
                cbar_kws={'label': 'Расстояние Махаланобиса'})
    ax.set_title('График 4. Попарные расстояния Махаланобиса между классами\n'
                 '(полный признаковый набор)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Класс'); ax.set_ylabel('Класс')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_04_maha_heatmap.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 4 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 5
def plot_bhatta_forward_selection(history: list, selected_names: list, out_dir: str):
    """
    [График 5] Кривая жадного отбора признаков (Бхаттачарья).

    Показывает, как накопленное расстояние Бхаттачарьи растёт по мере
    добавления каждого нового признака. «Колено» кривой указывает на
    оптимальный размер подмножества признаков.
    """
    if not history:
        print("  ⚠️  История Бхаттачарья пуста — график 5 пропущен")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(1, len(history) + 1))
    ax.plot(x, history, marker='o', linewidth=2.5, markersize=8,
            color='#E63946', markerfacecolor='white', markeredgewidth=2.5)

    # Подписи добавленных признаков
    for i, (xi, yi, name) in enumerate(zip(x, history, selected_names)):
        ax.annotate(name, xy=(xi, yi), xytext=(xi, yi + max(history)*0.04),
                    ha='center', fontsize=8, rotation=20,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    # Прирост на каждом шаге
    ax2 = ax.twinx()
    gains = [history[0]] + [history[i] - history[i-1] for i in range(1, len(history))]
    ax2.bar(x, gains, alpha=0.25, color='#E63946', label='Прирост D_B')
    ax2.set_ylabel('Прирост D_B', color='#E63946')
    ax2.tick_params(axis='y', labelcolor='#E63946')

    ax.set_xlabel('Количество признаков (шаг отбора)')
    ax.set_ylabel('Накопленное расстояние Бхаттачарьи D_B')
    ax.set_title('График 5. Forward Selection по критерию Бхаттачарьи\n'
                 '(жадный отбор, пара классов 2 и 11)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_05_forward_bhatta.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 5 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 6
def plot_ml_forward_selection(history: list, selected_names: list, out_dir: str):
    """
    [График 6] Кривая Forward Selection по точности kNN (5-fold CV).

    Позволяет сравнить с критерием Бхаттачарьи: оба метода должны
    отбирать схожие по информативности признаки.
    """
    if not history:
        print("  ⚠️  История ML пуста — график 6 пропущен")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(1, len(history) + 1))
    ax.plot(x, [v * 100 for v in history], marker='s', linewidth=2.5,
            markersize=8, color='#2A9D8F',
            markerfacecolor='white', markeredgewidth=2.5, label='Accuracy (5-fold CV)')

    for xi, yi, name in zip(x, history, selected_names):
        ax.annotate(name, xy=(xi, yi*100),
                    xytext=(xi, yi*100 + max(history)*3),
                    ha='center', fontsize=8, rotation=20,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    ax.set_xlabel('Количество признаков (шаг отбора)')
    ax.set_ylabel('Точность классификации, %')
    ax.set_title('График 6. Forward Selection по критерию kNN-Accuracy\n'
                 '(5-fold CV, k=5, все классы маски)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Зоны качества
    ax.axhspan(90, 105, alpha=0.07, color='green', label='Отлично (>90%)')
    ax.axhspan(70, 90,  alpha=0.07, color='yellow')
    ax.axhspan(0,  70,  alpha=0.07, color='red')

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_06_forward_ml.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 6 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 7
def plot_kde_ellipsoids(dataset: np.ndarray, mask: np.ndarray,
                         names: list, out_dir: str,
                         cls_pair=(2, 11)):
    """
    [График 7] Совместное распределение двух классов (KDE-эллипсоиды).

    Пространство проекции: Mean (ось X) vs Std (или Rho_Avg) (ось Y).
    Форма контуров плотности отражает геометрию ковариационных эллипсоидов,
    используемых в расстоянии Бхаттачарьи.
    """
    mask_flat = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])

    feat_x = 'Mean'    if 'Mean'    in names else names[0]
    feat_y = 'Rho_Avg' if 'Rho_Avg' in names else (names[1] if len(names) > 1 else names[0])
    ix, iy = names.index(feat_x), names.index(feat_y)

    fig, ax = plt.subplots(figsize=(8, 7))

    for cls in cls_pair:
        pixels = data_flat[mask_flat == cls]
        if len(pixels) < 5:
            continue
        color = PALETTE.get(cls, DEFAULT_COLORS[0])
        xv, yv = pixels[:, ix], pixels[:, iy]

        # Рисуем контуры плотности
        from scipy.stats import gaussian_kde
        try:
            xy  = np.vstack([xv, yv])
            kde = gaussian_kde(xy, bw_method=0.3)
            xmin, xmax = xv.min(), xv.max()
            ymin, ymax = yv.min(), yv.max()
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            z  = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax.contourf(xx, yy, z, levels=5, alpha=0.30, colors=[color]*5)
            ax.contour( xx, yy, z, levels=5, colors=[color], linewidths=1.5)
        except Exception:
            ax.scatter(xv[::20], yv[::20], alpha=0.3, s=10, color=color)

        # Центроид
        ax.scatter(np.mean(xv), np.mean(yv), marker='*', s=200,
                   color=color, edgecolors='black', zorder=5,
                   label=_safe_label(cls))

    ax.set_xlabel(f'Признак: {feat_x}', fontsize=11)
    ax.set_ylabel(f'Признак: {feat_y}', fontsize=11)
    ax.set_title('График 7. KDE-эллипсоиды рассеяния для двух классов\n'
                 f'(проекция: {feat_x} vs {feat_y})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_07_kde_ellipsoids.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 7 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 8
def plot_feature_histograms(dataset: np.ndarray, mask: np.ndarray,
                             names: list, out_dir: str,
                             cls_pair=(2, 11)):
    """
    [График 8] Гистограммы всех признаков для пары классов.

    Сравниваются распределения для каждого признака отдельно.
    Площадь пересечения гистограмм ~= вероятность ошибки классификации
    по одному признаку.
    """
    mask_flat = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])
    c         = len(names)
    cols      = 4
    rows      = (c + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()

    for i, feat_name in enumerate(names):
        ax = axes[i]
        for cls in cls_pair:
            pixels = data_flat[mask_flat == cls, i]
            if len(pixels) < 5:
                continue
            color = PALETTE.get(cls, DEFAULT_COLORS[0])
            ax.hist(pixels, bins=40, density=True, alpha=0.5, color=color,
                    label=f'C{cls}', edgecolor='none')
        ax.set_title(feat_name, fontsize=9, fontweight='bold')
        ax.set_xlabel('Значение', fontsize=7)
        ax.set_ylabel('Плотность', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Скрываем лишние оси
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'График 8. Гистограммы признаков для классов {cls_pair[0]} и {cls_pair[1]}\n'
                 f'(C{cls_pair[0]}: {CLASS_NAMES.get(cls_pair[0],"?")} | '
                 f'C{cls_pair[1]}: {CLASS_NAMES.get(cls_pair[1],"?")})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_08_feature_histograms.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 8 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 9
def plot_boxplot_by_class(dataset: np.ndarray, mask: np.ndarray,
                           names: list, out_dir: str,
                           max_classes: int = 6):
    """
    [График 9] Box-plot (ящики с усами) нормализованных признаков по классам.

    Позволяет сравнить медиану и разброс каждого признака сразу для
    всех классов. Признаки нормализованы в [0,1] для сопоставимости.
    """
    mask_flat = mask.flatten()
    data_flat = dataset.reshape(-1, dataset.shape[-1])
    classes   = sorted(c for c in np.unique(mask) if c > 0)[:max_classes]

    # Нормализация [0, 1]
    Xn = data_flat.copy().astype(np.float64)
    for i in range(Xn.shape[1]):
        mn, mx = Xn[:, i].min(), Xn[:, i].max()
        if mx > mn:
            Xn[:, i] = (Xn[:, i] - mn) / (mx - mn)

    n_feat = len(names)
    cols   = 2
    rows   = (n_feat + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 3.5))
    axes = axes.flatten()

    for i, feat_name in enumerate(names):
        ax    = axes[i]
        data_groups = []
        labels_bp   = []
        colors_bp   = []
        for cls in classes:
            pixels = Xn[mask_flat == cls, i]
            if len(pixels) < 5:
                continue
            data_groups.append(pixels)
            labels_bp.append(f'C{cls}')
            colors_bp.append(PALETTE.get(cls, DEFAULT_COLORS[classes.index(cls) % 10]))

        bp = ax.boxplot(data_groups, patch_artist=True,
                        notch=False, vert=True, widths=0.5,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(range(1, len(labels_bp) + 1))
        ax.set_xticklabels(labels_bp, fontsize=8)
        ax.set_title(feat_name, fontsize=9, fontweight='bold')
        ax.set_ylabel('Норм. значение [0,1]', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('График 9. Box-plot нормализованных признаков по классам\n'
                 '(медиана, IQR, выбросы)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_09_boxplot_classes.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 9 сохранён: {os.path.basename(path)}")


# --------------------------------------------------------------------------- GRAPH 10
def plot_selected_feature_map(dataset: np.ndarray, mask: np.ndarray,
                               names: list, selected_indices: list,
                               out_dir: str, max_maps: int = 4):
    """
    [График 10] Псевдоцветная карта отобранных признаков на снимке.

    Отображает пространственное распределение информативных признаков
    совместно с эталонной маской классов.
    """
    if not selected_indices:
        print("  ⚠️  Нет отобранных признаков — график 10 пропущен")
        return

    idxs  = selected_indices[:max_maps]
    n     = len(idxs)
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 5, 5))

    # Маска классов
    classes = sorted(c for c in np.unique(mask) if c > 0)
    cmap_mask = plt.cm.get_cmap('tab10', len(classes) + 1)
    axes[0].imshow(mask, cmap=cmap_mask, interpolation='nearest')
    axes[0].set_title('Эталонная маска\n(Ground Reference)', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for j, idx in enumerate(idxs):
        ax   = axes[j + 1]
        feat = dataset[:, :, idx]
        im   = ax.imshow(feat, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Признак: {names[idx]}\n(ранг {j+1})', fontsize=10, fontweight='bold')
        ax.axis('off')

    fig.suptitle('График 10. Пространственное распределение отобранных признаков\n'
                 '(результат Forward Selection)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_10_feature_maps.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✅ График 10 сохранён: {os.path.basename(path)}")


# ---------------------------------------------------------------------------
# ============================================================================
# ЧАСТЬ 8: ГЛАВНЫЙ ПАЙПЛАЙН
# ============================================================================
# ---------------------------------------------------------------------------

def main():
    """
    Основная функция — запускает полный пайплайн:

    1. Загрузка / генерация данных
    2. Вычисление признакового пространства
    3. Статистический анализ классов (Махаланобис, Бхаттачарья)
    4. Forward Selection (статистический и ML)
    5. Построение 10 графиков
    6. Вывод итоговых таблиц
    """
    print("=" * 70)
    print("  Курсовая работа: Статистические методы отбора признаков")
    print("  при классификации космических изображений (MultiSenGE)")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Директории вывода
    # -----------------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir      = os.path.join(project_root, 'output')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📁 Графики будут сохранены в: {out_dir}")

    # -----------------------------------------------------------------------
    # Шаг 1. Загрузка данных
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 1: Загрузка данных")
    print("─" * 60)

    img_path, mask_path = None, None

    if HAS_RASTERIO:
        img_path, mask_path = auto_find_data(project_root)

    if img_path and mask_path and HAS_RASTERIO:
        print(f"  Снимок:  {os.path.basename(img_path)}")
        print(f"  Маска:   {os.path.basename(mask_path)}")
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.uint8)
            with rasterio.open(img_path) as src:
                img_data = src.read().astype(np.float32)

            # Если многоканальный — берём среднее для одноканальной обработки
            if img_data.ndim == 3 and img_data.shape[0] > 1:
                img_gray = np.mean(img_data, axis=0)
            else:
                img_gray = img_data.squeeze()

            # Нормализация 0–255
            mn, mx = img_gray.min(), img_gray.max()
            img_norm = ((img_gray - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)

            if mask.shape != img_norm.shape:
                mask = zoom(mask,
                            (img_norm.shape[0] / mask.shape[0],
                             img_norm.shape[1] / mask.shape[1]),
                            order=0).astype(np.uint8)
            use_demo = False
        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}. Переключаюсь на демо-данные.")
            use_demo = True
    else:
        use_demo = True

    if use_demo:
        print("  🎲 Генерация синтетических данных (256×256)...")
        img_norm, mask = _generate_synthetic_data()
        print(f"  Размер снимка: {img_norm.shape}, Маска: {mask.shape}")

    unique_cls = np.unique(mask)
    unique_cls = unique_cls[unique_cls > 0]
    print(f"  Классы в маске: {unique_cls.tolist()}")

    # -----------------------------------------------------------------------
    # Шаг 2. Вычисление признакового пространства
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 2: Вычисление признаков (окно 15×15)")
    print("─" * 60)

    feat_dict = extract_all_features(img_norm, window_size=15, is_multispectral=False)
    dataset, names = make_feature_sandwich(feat_dict)
    print(f"\n  📦 Признаковый куб: {dataset.shape}")
    print(f"  📋 Признаки ({len(names)}): {names}")

    # -----------------------------------------------------------------------
    # Шаг 3. Статистический анализ классов
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 3: Анализ классов — расстояния Махаланобиса и Бхаттачарьи")
    print("─" * 60)

    stats = {}
    for cls in unique_cls:
        mean_v, cov_m = calculate_class_stats(dataset, mask, int(cls))
        if mean_v is not None:
            stats[int(cls)] = {'mean': mean_v, 'cov': cov_m}

    classes_list = sorted(stats.keys())
    print(f"\n  Классов обработано: {len(classes_list)}")

    df_maha, df_bhatt = compute_all_pairwise_distances(stats, classes_list)

    # Текстовый отчёт
    print("\n  Попарные расстояния (Бхаттачарья):")
    print(df_bhatt.round(3).to_string())
    print("\n  Попарные расстояния (Махаланобис):")
    print(df_maha.round(3).to_string())

    # -----------------------------------------------------------------------
    # Шаг 4. Forward Selection — Бхаттачарья
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 4: Forward Selection по критерию Бхаттачарьи (пара 2↔11)")
    print("─" * 60)

    # Выбираем пару классов: 2 и 11 (если есть), иначе — первые два
    if 2 in unique_cls and 11 in unique_cls:
        pair = (2, 11)
    elif len(unique_cls) >= 2:
        pair = (int(unique_cls[0]), int(unique_cls[1]))
    else:
        pair = None

    sel_bhatta, hist_bhatta = [], []
    if pair:
        sel_bhatta, hist_bhatta = forward_selection_bhatta(
            dataset, mask, target_classes=pair, eps=0.001, max_features=10)
        sel_bhatta_names = [names[i] for i in sel_bhatta]
        print(f"\n  🏆 Отобрано признаков (Бхаттачарья): {sel_bhatta_names}")
    else:
        sel_bhatta_names = []
        print("  ⚠️  Недостаточно классов для Бхаттачарья-отбора")

    # -----------------------------------------------------------------------
    # Шаг 5. Forward Selection — ML (kNN)
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 5: Forward Selection по точности kNN (5-fold CV)")
    print("─" * 60)

    target_cls_ml = [int(c) for c in unique_cls if c > 0]
    sel_ml, hist_ml = forward_selection_ml(
        dataset, mask, target_classes=target_cls_ml,
        eps=0.001, max_features=10)
    sel_ml_names = [names[i] for i in sel_ml]
    print(f"\n  🏆 Отобрано признаков (kNN): {sel_ml_names}")

    # -----------------------------------------------------------------------
    # Шаг 6. Визуализация (10 графиков)
    # -----------------------------------------------------------------------
    print("\n" + "─" * 60)
    print("ШАГ 6: Построение графиков")
    print("─" * 60)

    plot_feature_correlation(dataset, mask, names, out_dir)
    plot_class_distributions(dataset, mask, names, out_dir)
    plot_bhatta_heatmap(df_bhatt, out_dir)
    plot_maha_heatmap(df_maha, out_dir)
    plot_bhatta_forward_selection(hist_bhatta, sel_bhatta_names, out_dir)
    plot_ml_forward_selection(hist_ml, sel_ml_names, out_dir)
    plot_kde_ellipsoids(dataset, mask, names, out_dir, cls_pair=pair if pair else (int(unique_cls[0]), int(unique_cls[0])))
    plot_feature_histograms(dataset, mask, names, out_dir, cls_pair=pair if pair else (int(unique_cls[0]), int(unique_cls[0])))
    plot_boxplot_by_class(dataset, mask, names, out_dir)
    plot_selected_feature_map(dataset, mask, names, sel_bhatta if sel_bhatta else sel_ml, out_dir)

    # -----------------------------------------------------------------------
    # Итоговый отчёт в консоль
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 70)

    print(f"\n📐 Признаковое пространство:")
    print(f"   Всего признаков:    {len(names)}")
    print(f"   Размер куба:        {dataset.shape}")
    print(f"   Классов в маске:    {len(classes_list)}")

    if sel_bhatta_names:
        print(f"\n📊 Forward Selection (Бхаттачарья) — классы {pair}:")
        for rank, name in enumerate(sel_bhatta_names, 1):
            d = hist_bhatta[rank-1] if rank-1 < len(hist_bhatta) else '—'
            print(f"   {rank}. {name:<18s}  D_B = {d:.4f}" if isinstance(d, float) else f"   {rank}. {name}")

    if sel_ml_names:
        print(f"\n🤖 Forward Selection (kNN) — все классы:")
        for rank, name in enumerate(sel_ml_names, 1):
            acc = hist_ml[rank-1]*100 if rank-1 < len(hist_ml) else 0
            print(f"   {rank}. {name:<18s}  Accuracy = {acc:.1f}%")

    # Пересечение двух наборов
    common = set(sel_bhatta_names) & set(sel_ml_names)
    if common:
        print(f"\n✨ Признаки, отобранные обоими методами: {sorted(common)}")
        print("   (Высокая согласованность методов подтверждает их информативность)")

    print(f"\n📂 Все графики сохранены в: {out_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()