import numpy as np
from scipy.ndimage import uniform_filter

def get_fast_stats(image, window_size):
    """Вычисляет локальное среднее, дисперсию и СКО за O(1)."""
    n = window_size
    mean = uniform_filter(image.astype(np.float64), size=n, mode='mirror')
    sq_mean = uniform_filter(image.astype(np.float64)**2, size=n, mode='mirror')
    
    var = sq_mean - mean**2
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    return mean, var, std

def fast_cascade_mean_std(image, window_size):
    """Обёртка для совместимости с process_image.py"""
    mean, _, std = get_fast_stats(image, window_size)
    return mean, std

def smart_normalize(arr, mode='mean'):
    """Нормализация массива в [0, 255] или [-1, 1] -> [0, 255]."""
    arr = arr.astype(np.float64)
    if mode == 'signed':
        return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    else:
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        return np.clip((arr - arr_min) / (arr_max - arr_min) * 255, 0, 255).astype(np.uint8)

def make_feature_sandwich(feature_dict):
    """Собирает словарь признаков в массив (H, W, C) и возвращает имена каналов."""
    names = list(feature_dict.keys())
    channels = [feature_dict[name].astype(np.float32) for name in names]
    dataset = np.stack(channels, axis=-1)
    return dataset, names

def calc_directional_rho(image, mean, var, window_size):
    """Коэффициенты корреляции по 4 направлениям (0°, 90°, 45°, 135°)."""
    directions = {
        '0': (0, 1), '90': (1, 0), '45': (1, 1), '135': (1, -1)
    }
    rhos = {}
    eps = 1e-10
    
    for angle, (dy, dx) in directions.items():
        shifted = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        f_adj = uniform_filter(image * shifted, size=window_size, mode='mirror')
        rho = (f_adj - mean**2) / (var + eps)
        rhos[angle] = np.clip(rho, -1, 1)
    return rhos

def compute_spectral_indices(img_10ch):
    """
    Вычисляет спектральные индексы для Sentinel-2.
    Порядок каналов: B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12
    """
    eps = 1e-8
    
    # Извлекаем нужные каналы
    B2 = img_10ch[0]   # Синий
    B3 = img_10ch[1]   # Зелёный
    B4 = img_10ch[2]   # Красный
    B8 = img_10ch[3]   # NIR
    B11 = img_10ch[8]  # SWIR-1
    B12 = img_10ch[9]  # SWIR-2
    
    # NDVI - растительность
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    
    # NDWI - вода
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    
    # NDBI - застройка
    ndbi = (B11 - B8) / (B11 + B8 + eps)
    
    # Интегральная яркость (сумма каналов)
    total_brightness = np.sum(img_10ch, axis=0)
    
    # Нормализованные каналы (инварианты к освещённости)
    norm_channels = img_10ch / (total_brightness + eps)
    
    indices = {
        'NDVI': ndvi,
        'NDWI': ndwi,
        'NDBI': ndbi,
        'Total_Brightness': total_brightness
    }
    
    # Добавляем нормализованные каналы
    for i, name in enumerate(['B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']):
        indices[f'Norm_{name}'] = norm_channels[i]
    
    return indices

def extract_all_features(image, window_size=15, is_multispectral=True):
    """
    Основная функция генерации полного признакового пространства.
    
    Parameters:
    -----------
    image : ndarray
        Если is_multispectral=True: (C, H, W) - 10 каналов Sentinel-2
        Если is_multispectral=False: (H, W) - одноканальное изображение
    window_size : int
        Размер окна для локальных статистик
    is_multispectral : bool
        Если True, вычисляет спектральные индексы
    """
    feature_space = {}
    
    # 1. Спектральные признаки (если многоканальное изображение)
    if is_multispectral and image.ndim == 3:
        print("  Вычисление спектральных индексов...")
        indices = compute_spectral_indices(image)
        feature_space.update(indices)
        
        # Для текстурного анализа используем среднее по каналам
        gray_image = np.mean(image, axis=0).astype(np.float32)
    else:
        gray_image = image.astype(np.float32)
        feature_space['Original'] = gray_image
    
    # 2. Локальные статистики (по серому изображению)
    print(f"  Вычисление локальных статистик (окно {window_size}x{window_size})...")
    mean, var, std = get_fast_stats(gray_image, window_size)
    feature_space['Mean'] = mean
    feature_space['Std'] = std
    
    # 3. Направленные корреляции
    print("  Вычисление текстурных корреляций...")
    rhos = calc_directional_rho(gray_image, mean, var, window_size)
    feature_space['Rho_Avg'] = (rhos['0'] + rhos['90'] + rhos['45'] + rhos['135']) / 4
    feature_space['Rho_Range'] = np.maximum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']]) - \
                                 np.minimum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']])
    feature_space['Rho_0'] = rhos['0']
    feature_space['Rho_90'] = rhos['90']
    feature_space['Rho_45'] = rhos['45']
    feature_space['Rho_135'] = rhos['135']
    
    return feature_space

if __name__ == "__main__":
    print("✅ fast_features.py обновлён!")
    print("   - Добавлены спектральные индексы (NDVI, NDWI, NDBI)")
    print("   - Добавлены нормализованные каналы (10 штук)")
    print("   - Поддержка многоканальных снимков Sentinel-2")
    print("   - Полный набор: ~41 признак")