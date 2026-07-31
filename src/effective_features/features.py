"""
features.py - загрузка снимков и вычисление признаков.

Какие снимки читать и какие каналы в них лежат, берётся из dataset.json
(см. dataset.py), поэтому число признаков зависит от данных. На MultiSenGE
получается 41: 9 спектральных (нормализованные каналы + NDVI/NDWI/NDBI)
и 32 текстурных - 8 штук на каждое окно {3,5,7,9}.

Посчитанные признаки складываются в cache/, чтобы не считать их заново
при каждом прогоне.

Первичные признаки (Сумма, Сумма², произведения, Min, Max) используются
только как СЫРЬЁ и в матрицу отбора НЕ входят.
"""

import os
import random

import numpy as np
from scipy.ndimage import uniform_filter

from .config import ExperimentConfig, CLASS_NAMES

try:
    import rasterio
    from scipy.ndimage import zoom
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def get_fast_stats(image, window_size):
    """
    Локальные среднее (Mean) и дисперсия (Var) за O(1) на пиксель
    через интегральное изображение (uniform_filter).

    Var = E[I²] − (E[I])²   - формула (5) НИР.
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


def compute_spectral_features(image, spec):
    """
    Спектральные признаки: нормализованные каналы + индексы.

    Что именно считается, зависит от описания датасета:
      - в признаки идут каналы из spec.bands_for_features()
      - нормировка: Norm_Bi = Bi / сумма выбранных каналов
      - индексы считаются только те, для которых заданы нужные роли

    Раньше номера каналов и формулы были прописаны прямо здесь, поэтому
    работал ровно один датасет. Теперь всё берётся из dataset.json.
    """
    eps = 1e-8
    out = {}

    band_names = spec.bands_for_features()
    idx = [spec.band_order.index(b) for b in band_names]
    base = np.stack([image[i] for i in idx], axis=0)
    total = np.sum(base, axis=0)                  # S - сырьё, в признаки не идёт

    for i, name in enumerate(band_names):
        out[f'Norm_{name}'] = (base[i] / (total + eps)).astype(np.float32)

    def band(role):
        """Канал по роли. None, если роль не задана в описании."""
        i = spec.band_index(role)
        return None if i is None else image[i]

    def ratio(a, b):
        """Нормализованная разность (a − b) / (a + b) - общая форма
        для NDVI, NDWI и NDBI."""
        return ((a - b) / (a + b + eps)).astype(np.float32)

    nir, red, green, swir1 = band('nir'), band('red'), band('green'), band('swir1')

    if nir is not None and red is not None:
        out['NDVI'] = ratio(nir, red)
    if green is not None and nir is not None:
        out['NDWI'] = ratio(green, nir)
    if swir1 is not None and nir is not None:
        out['NDBI'] = ratio(swir1, nir)

    return out


def extract_all_features(image, cfg: ExperimentConfig):
    """
    Собирает признаки из наборов, указанных в настройках.

    Раньше состав был прописан прямо здесь. Теперь наборы лежат в реестре
    (feature_sets.py), и исследователь может добавить свой, не трогая
    этот код: достаточно положить файл в папку user_features.
    """
    from . import feature_sets as fsets

    fsets.load_user_sets(cfg.project_root)

    spec = cfg.spec if image.ndim == 3 else None

    # Яркость для текстурных признаков - среднее по тем каналам,
    # что идут в признаки. Если описания нет, берём все.
    if spec is not None and cfg.use_spectral:
        idx = [spec.band_order.index(b) for b in spec.bands_for_features()]
        gray = np.mean(np.stack([image[i] for i in idx], axis=0),
                       axis=0).astype(np.float32)
    else:
        gray = (np.mean(image, axis=0) if image.ndim == 3 else image).astype(np.float32)

    ctx = fsets.FeatureContext(
        image=image, gray=gray, spec=spec,
        window_sizes=tuple(cfg.window_sizes),
    )

    fs = {}
    for set_id in cfg.active_feature_sets():
        fset = fsets.get(set_id)
        if fset is None:
            print(f"    Набор «{set_id}» не найден - пропускаю")
            continue
        if not fset.is_available(ctx):
            print(f"    Набор «{fset.name}» пропущен: нужен многоканальный снимок")
            continue

        try:
            produced = fset.compute(ctx)
        except Exception as e:
            # Ошибка в пользовательском наборе не должна ронять весь расчёт
            print(f"    Набор «{fset.name}» не посчитался: {type(e).__name__}: {e}")
            continue

        clash = set(produced) & set(fs)
        if clash:
            print(f"    Набор «{fset.name}»: имена уже заняты, пропускаю их - "
                  f"{', '.join(sorted(clash))}")
            produced = {k: v for k, v in produced.items() if k not in clash}

        print(f"    {fset.name}: {len(produced)} признаков")
        fs.update(produced)

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


def load_pair(img_name, mask_name, spec):
    """Загружает снимок (C,H,W) float32 и маску (H,W) uint8.
    Пути берутся из описания датасета."""
    with rasterio.open(os.path.join(spec.images_path, img_name.strip())) as s:
        image = s.read().astype(np.float32)
    with rasterio.open(os.path.join(spec.masks_path, mask_name.strip())) as s:
        mask = s.read(1).astype(np.uint8)
    return image, mask


def _read_mask_classes(mask_name, spec):
    """Быстро читает только маску патча и возвращает множество классов
    в ней (без нулей = фон). Используется только для прореживания -
    признаки при этом НЕ считаются, так что это дёшево."""
    with rasterio.open(os.path.join(spec.masks_path, mask_name.strip())) as f:
        mask = f.read(1)
    return set(np.unique(mask).tolist()) - {0}


def select_pairs_thinned(pairs, cfg: ExperimentConfig, spec):
    """
    Систематическое прореживание: берём каждый k-й патч по порядку.

    Случайность здесь не нужна - выборка получается одна и та же при любом
    запуске, а покрытие датасета равномерное по построению.

    Проблема чистого прореживания: самые редкие классы (напр. Торфяники -
    ~8 тыс. пикселей на весь датасет из 1911 патчей, Хвойные леса - ~16 тыс.)
    могут не попасть ни в один патч выборки просто по шагу. Поэтому после
    систематического отбора мы сканируем маски (только слой разметки,
    без вычисления признаков - быстро) и явно добираем патчи, которые
    закрывают недостающие классы.
    """
    n_total = len(pairs)
    target = cfg.thinning_target_patches
    step = max(1, n_total // target)
    thinned = list(pairs[::step][:target])
    thinned_set = set(thinned)

    print(f"\n  Прореживание: шаг {step}, взято {len(thinned)} патчей "
          f"из {n_total} (равномерно по датасету)")
    print("  Проверка покрытия классов (сканирую маски, без признаков)...")

    covered = set()
    for img, msk in thinned:
        covered |= _read_mask_classes(msk, spec)

    all_classes = set(cfg.class_names().keys())
    missing = all_classes - covered
    if missing:
        print(f"  Не хватает классов: {sorted(missing)} - ищу патчи, где они есть...")
        for img, msk in pairs:
            if not missing:
                break
            if (img, msk) in thinned_set:
                continue
            classes_here = _read_mask_classes(msk, spec)
            found = classes_here & missing
            if found:
                thinned.append((img, msk))
                thinned_set.add((img, msk))
                missing -= found
                print(f"    + добавлен {img} (закрывает классы {sorted(found)})")
        if missing:
            print(f"  ВНИМАНИЕ: классы {sorted(missing)} не найдены ни в одном патче датасета.")

    print(f"  Итог прореживания: {len(thinned)} патчей, "
          f"классы покрыты: {sorted(all_classes - missing)}")
    return thinned


def select_pairs(cfg: ExperimentConfig):
    """
    cfg.use_thinning=True → систематическое прореживание с гарантией всех
                            классов, выборка зависит от сида
                            (см. select_pairs_thinned);
    cfg.n_patches=None    → весь датасет;
    иначе                 → случайные n_patches патчей.

    Сами пары снимок-маска берутся из описания датасета: либо по спискам
    файлов, либо сопоставлением по именам.
    """
    from .dataset import find_pairs
    spec = cfg.spec
    pairs = find_pairs(spec)

    if cfg.use_thinning:
        return select_pairs_thinned(pairs, cfg, spec)

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


def _patch_cache_path(cfg, img_name):
    """
    Путь к кеш-файлу признаков патча.
    Имя кодирует и патч, и конфигурацию признаков (окна/спектральность),
    чтобы при смене параметров не подхватился несовместимый кеш.
    """
    base = os.path.splitext(os.path.basename(img_name.strip()))[0]
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


def get_patch_features(cfg, img_name, raw_img):
    """
    Возвращает (cube, names) признаков патча, используя кеш по правилам:
      force_recompute=True → считаем заново, обновляем кеш (если save_cache);
      use_cache=True       → пробуем прочитать из кеша, иначе считаем;
      иначе                → считаем на лету.
    """
    cache_path = _patch_cache_path(cfg, img_name)

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
    Какие данные читать и как их разбирать - берётся из dataset.json.
    Признаки патчей кешируются (см. cfg.use_cache / save_cache / force_recompute).
    """
    if not HAS_RASTERIO:
        raise RuntimeError("rasterio не установлен. pip install rasterio")

    spec = cfg.spec
    print(f"  Датасет: {spec.name}")

    pairs = select_pairs(cfg)

    cache_mode = ('пересчёт+кеш' if cfg.force_recompute else
                  'кеш' if cfg.use_cache else 'без кеша')
    print(f"  Режим признаков: {cache_mode}")

    n_from_cache = n_computed = 0
    names = None
    X_parts, y_parts = [], []   # копим куски, склеиваем один раз в конце
    total_pixels = 0
    for idx, (img_name, mask_name) in enumerate(pairs, 1):
        print(f"\n  Патч {idx}/{len(pairs)}: {img_name}")
        try:
            cache_path = _patch_cache_path(cfg, img_name)
            cached = (cfg.use_cache and not cfg.force_recompute
                      and os.path.isfile(cache_path))

            # Маску надо прочитать всегда (она не кешируется - лёгкая)
            raw_img, patch_mask = load_pair(img_name, mask_name, spec)

            cube, patch_names = get_patch_features(cfg, img_name, raw_img)
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
            X_parts.append(X_patch)
            y_parts.append(y_patch)
            total_pixels += len(X_patch)
            print(f"  +{len(X_patch):,} пкс | итого: {total_pixels:,}")

        except Exception as e:
            print(f"  Ошибка патча {img_name}: {e}")

    print(f"\n  Признаки: из кеша {n_from_cache}, посчитано {n_computed}")

    if not X_parts or total_pixels < 100:
        raise RuntimeError("Данные не загружены. Проверьте data/ и lists/.")

    X_global = np.concatenate(X_parts, axis=0)
    y_global = np.concatenate(y_parts, axis=0)

    return X_global, y_global, names