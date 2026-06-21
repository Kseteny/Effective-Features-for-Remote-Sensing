"""
features.py — вычисление базового признакового пространства (41 признак)
и загрузка данных датасета MultiSenGE (Sentinel-2).

Базовый набор (41):
  9 спектральных   — 6 нормализованных каналов + NDVI/NDWI/NDBI
  32 текстурных    — 8 производных × 4 окна {3,5,7,9}:
                     Mean, Var, Rho_0, Rho_90, Rho_45, Rho_135, Rho_Avg, Rho_Range

Первичные признаки (Сумма, Сумма², произведения, Min, Max) используются
только как СЫРЬЁ и в матрицу отбора НЕ входят.
"""

import os
import random

import numpy as np
from scipy.ndimage import uniform_filter

from .config import ExperimentConfig

try:
    import rasterio
    from scipy.ndimage import zoom
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ===========================================================================
# ВЫЧИСЛЕНИЕ ПРИЗНАКОВ
# ===========================================================================

def get_fast_stats(image, window_size):
    """
    Локальные среднее (Mean) и дисперсия (Var) за O(1) на пиксель
    через интегральное изображение (uniform_filter).

    Var = E[I²] − (E[I])²   — формула (5) НИР.
    Возвращает (mean, var).
    """
    img64   = image.astype(np.float64)
    mean    = uniform_filter(img64,      size=window_size, mode='mirror')
    sq_mean = uniform_filter(img64 ** 2, size=window_size, mode='mirror')
    var     = np.maximum(sq_mean - mean ** 2, 0.0)
    return mean, var


def calc_directional_rho(image, mean, var, window_size):
    """
    Направленные коэффициенты корреляции: 0°, 90°, 45°, 135°.

    ρ_dir = (E[I · shift_dir(I)] − μ²) / (σ² + ε)

    Высокое ρ → упорядоченная текстура, низкое → хаотичная.
    """
    eps  = 1e-10
    dirs = {'0': (0, 1), '90': (1, 0), '45': (1, 1), '135': (1, -1)}
    rhos = {}
    for angle, (dy, dx) in dirs.items():
        shifted = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        f_adj   = uniform_filter(image * shifted, size=window_size, mode='mirror')
        rhos[angle] = np.clip((f_adj - mean ** 2) / (var + eps), -1.0, 1.0)
    return rhos


def compute_spectral_features(img_10ch):
    """
    9 спектральных признаков:
      - 6 нормализованных каналов: Norm_Bi = Bi / ΣBj
      - 3 индекса: NDVI, NDWI, NDBI

    Сырые каналы и интегральная яркость S — только сырьё, в признаки не идут.
    Порядок входных каналов: B2 B3 B4 B8 B5 B6 B7 B8A B11 B12 (индексы 0..9).
    """
    eps = 1e-8
    B2, B3, B4, B8 = img_10ch[0], img_10ch[1], img_10ch[2], img_10ch[3]
    B11, B12       = img_10ch[8], img_10ch[9]

    base  = np.stack([B2, B3, B4, B8, B11, B12], axis=0)
    total = np.sum(base, axis=0)                  # S — сырьё

    out = {}
    for i, name in enumerate(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']):
        out[f'Norm_{name}'] = (base[i] / (total + eps)).astype(np.float32)

    out['NDVI'] = ((B8  - B4)  / (B8  + B4  + eps)).astype(np.float32)
    out['NDWI'] = ((B3  - B8)  / (B3  + B8  + eps)).astype(np.float32)
    out['NDBI'] = ((B11 - B8)  / (B11 + B8  + eps)).astype(np.float32)
    return out  # 9 признаков


def extract_all_features(image, cfg: ExperimentConfig):
    """
    Полное базовое признаковое пространство (41 признак при дефолтном cfg).

      Спектральные (9, если cfg.use_spectral)
      Текстурные   (8 × len(cfg.window_sizes))
    """
    fs = {}

    if cfg.use_spectral and image.ndim == 3:
        print("    Спектральные признаки (6 нормализованных + NDVI/NDWI/NDBI)...")
        fs.update(compute_spectral_features(image))
        gray = np.mean(np.stack([image[i] for i in [0, 1, 2, 3, 8, 9]], axis=0),
                       axis=0).astype(np.float32)
    else:
        gray = (np.mean(image, axis=0) if image.ndim == 3 else image).astype(np.float32)

    for w in cfg.window_sizes:
        print(f"    Текстурные признаки (окно {w}×{w})...")
        mean, var = get_fast_stats(gray, w)
        fs[f'Mean_{w}'] = mean.astype(np.float32)
        fs[f'Var_{w}']  = var.astype(np.float32)

        rhos = calc_directional_rho(gray, mean, var, w)
        fs[f'Rho_Avg_{w}']   = ((rhos['0'] + rhos['90'] + rhos['45'] + rhos['135']) / 4
                                 ).astype(np.float32)
        fs[f'Rho_Range_{w}'] = (np.maximum.reduce(list(rhos.values())) -
                                np.minimum.reduce(list(rhos.values()))).astype(np.float32)
        for a in ('0', '90', '45', '135'):
            fs[f'Rho_{a}_{w}'] = rhos[a].astype(np.float32)
    return fs


def make_feature_sandwich(fd):
    """dict → (H, W, C) float32 + список имён каналов."""
    names = list(fd.keys())
    return np.stack([fd[n].astype(np.float32) for n in names], axis=-1), names


def parse_feature_window(name):
    """'Rho_Avg_7' → 7; 'NDVI' → None."""
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


# ===========================================================================
# ЗАГРУЗКА ДАННЫХ (MultiSenGE)
# ===========================================================================

def load_pair(s2_name, gr_name, data_dir):
    """Загружает (10,H,W) float32 снимок и (H,W) uint8 маску."""
    with rasterio.open(os.path.join(data_dir, 's2_pref', s2_name.strip())) as s:
        image = s.read().astype(np.float32)
    with rasterio.open(os.path.join(data_dir, 'ground_reference', gr_name.strip())) as s:
        mask = s.read(1).astype(np.uint8)
    return image, mask


def get_file_lists(lists_dir):
    """Читает out_s2_pref.txt / out_gr_pref.txt."""
    with open(os.path.join(lists_dir, 'out_s2_pref.txt')) as f:
        s2 = [l.strip() for l in f if l.strip()]
    with open(os.path.join(lists_dir, 'out_gr_pref.txt')) as f:
        gr = [l.strip() for l in f if l.strip()]
    return s2, gr


def select_pairs(cfg: ExperimentConfig, lists_dir):
    """cfg.n_patches=None → весь датасет; иначе случайные N патчей."""
    s2_list, gr_list = get_file_lists(lists_dir)
    if len(s2_list) != len(gr_list):
        raise ValueError("Количество снимков и масок не совпадает")

    pairs = list(zip(s2_list, gr_list))
    if cfg.n_patches is None:
        print(f"\n  Используется ВЕСЬ датасет: {len(pairs)} патчей")
        return pairs

    rng = random.Random(cfg.random_seed)
    rng.shuffle(pairs)
    selected = pairs[:cfg.n_patches]
    print(f"\n  Выбрано случайных патчей: {len(selected)} из {len(pairs)}")
    return selected


def build_global_dataset(feature_cube, mask):
    """(H,W,C) → (X, y) только размеченных пикселей."""
    c = feature_cube.shape[-1]
    X = feature_cube.reshape(-1, c)
    y = mask.flatten()
    valid = y > 0
    return X[valid], y[valid]


def subsample_dataset(X, y, cfg: ExperimentConfig):
    """Опциональная субдискретизация (если cfg.max_pixels_total задан)."""
    if cfg.max_pixels_total is None or len(X) <= cfg.max_pixels_total:
        return X, y
    rng = np.random.default_rng(cfg.random_seed)
    idx = rng.choice(len(X), size=cfg.max_pixels_total, replace=False)
    return X[idx], y[idx]


def rebuild_feature_cube(X, y):
    """Псевдо-куб (N,1,C) + маска (N,1) для совместимости с визуализацией."""
    n, c = X.shape
    return X[:, np.newaxis, :], y.reshape(n, 1)


def normalize_channels(raw_img):
    """Поканальная нормировка снимка в [0,255]."""
    img = np.moveaxis(raw_img, 0, -1).astype(np.float32)
    for ch in range(img.shape[-1]):
        mn, mx = img[:, :, ch].min(), img[:, :, ch].max()
        img[:, :, ch] = (img[:, :, ch] - mn) / (mx - mn + 1e-8) * 255
    return np.moveaxis(img, -1, 0)


def _patch_cache_path(cfg, s2_name):
    """
    Путь к кеш-файлу признаков патча.
    Имя кодирует и патч, и конфигурацию признаков (окна/спектральность),
    чтобы при смене параметров не подхватился несовместимый кеш.
    """
    base = os.path.splitext(os.path.basename(s2_name.strip()))[0]
    return os.path.join(cfg.cache_dir, f"{base}__{cfg.cache_key()}.npz")


def _compute_patch_features(raw_img, cfg):
    """Вычисление куба признаков патча (без кеша)."""
    if raw_img.ndim == 3 and cfg.use_spectral:
        img_norm = normalize_channels(raw_img)
        feat_dict = extract_all_features(img_norm, cfg)
    else:
        gray = np.mean(raw_img, axis=0) if raw_img.ndim == 3 else raw_img.squeeze()
        mn, mx = gray.min(), gray.max()
        gray = ((gray - mn) / (mx - mn + 1e-8) * 255).astype(np.float32)
        feat_dict = extract_all_features(gray, cfg)
    return make_feature_sandwich(feat_dict)   # (cube, names)


def get_patch_features(cfg, s2_name, raw_img):
    """
    Возвращает (cube, names) признаков патча, используя кеш по правилам:
      force_recompute=True → считаем заново, обновляем кеш (если save_cache);
      use_cache=True       → пробуем прочитать из кеша, иначе считаем;
      иначе                → считаем на лету.
    """
    cache_path = _patch_cache_path(cfg, s2_name)

    # 1) Чтение из кеша (если разрешено и не форсим пересчёт)
    if cfg.use_cache and not cfg.force_recompute and os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cube = data['cube']
            names = list(data['names'])
            print(f"    [кеш] признаки загружены из {os.path.basename(cache_path)}")
            return cube, names
        except Exception as e:
            print(f"    [кеш] повреждён ({e}), пересчитываю")

    # 2) Вычисление
    cube, names = _compute_patch_features(raw_img, cfg)

    # 3) Сохранение в кеш
    if cfg.save_cache:
        try:
            np.savez_compressed(cache_path, cube=cube, names=np.array(names, dtype=object))
        except Exception as e:
            print(f"    [кеш] не удалось сохранить: {e}")

    return cube, names


def load_all_data(cfg: ExperimentConfig):
    """
    Полный цикл загрузки: патчи → признаки → объединённая выборка (X, y, names).
    Читает РЕАЛЬНЫЙ датасет MultiSenGE из data/ и lists/.
    Признаки патчей кешируются (см. cfg.use_cache / save_cache / force_recompute).
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio не установлен. pip install rasterio")

    data_dir  = os.path.join(cfg.project_root, 'data')
    lists_dir = os.path.join(cfg.project_root, 'lists')

    has_lists = (os.path.isfile(os.path.join(lists_dir, 'out_s2_pref.txt')) and
                 os.path.isfile(os.path.join(lists_dir, 'out_gr_pref.txt')))
    if not has_lists:
        raise RuntimeError("Не найдены lists/out_s2_pref.txt и out_gr_pref.txt")

    pairs = select_pairs(cfg, lists_dir)

    cache_mode = ('пересчёт+кеш' if cfg.force_recompute else
                  'кеш' if cfg.use_cache else 'без кеша')
    print(f"  Режим признаков: {cache_mode}")

    n_from_cache = n_computed = 0
    X_global = y_global = names = None
    for idx, (s2_name, gr_name) in enumerate(pairs, 1):
        print(f"\n  Патч {idx}/{len(pairs)}: {s2_name}")
        try:
            cache_path = _patch_cache_path(cfg, s2_name)
            cached = (cfg.use_cache and not cfg.force_recompute
                      and os.path.isfile(cache_path))

            # Маску надо прочитать всегда (она не кешируется — лёгкая)
            raw_img, patch_mask = load_pair(s2_name, gr_name, data_dir)

            cube, patch_names = get_patch_features(cfg, s2_name, raw_img)
            if cached:
                n_from_cache += 1
            else:
                n_computed += 1

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
                X_global, y_global = X_patch, y_patch
            else:
                X_global = np.vstack([X_global, X_patch])
                y_global = np.concatenate([y_global, y_patch])
            print(f"  +{len(X_patch):,} пкс | итого: {len(X_global):,}")

        except Exception as e:
            print(f"  Ошибка патча {s2_name}: {e}")

    print(f"\n  Признаки: из кеша {n_from_cache}, посчитано {n_computed}")

    if X_global is None or len(X_global) < 100:
        raise RuntimeError("Данные не загружены. Проверьте data/ и lists/.")

    return X_global, y_global, names