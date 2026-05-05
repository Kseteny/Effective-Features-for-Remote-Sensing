import numpy as np
from skimage.feature import graycomatrix, graycoprops

def get_integral_image(img):
    """
    Создает интегральное изображение для быстрого суммирования.
    Это основа для оптимизации O(1).
    """
    return np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)

def get_window_sum(integral_img, L):
    """
    Вычисляет сумму в окне L x L за константное время.
    Формула: S = A + D - B - C
    """
    pad = L // 2
    # Создаем расширенное полотно для корректной обработки краев
    h, w = integral_img.shape
    padded = np.pad(integral_img, ((pad + 1, pad), (pad + 1, pad)), mode='edge')
    
    # Вычисляем сумму через углы прямоугольника
    res = (padded[L:, L:] + padded[:-L, :-L] - 
           padded[L:, :-L] - padded[:-L, L:])
    return res

def compute_basic_features(channel_img, L):
    """
    Вычисляет среднее и дисперсию в скользящем окне L x L.
    Реализовано через интегральные изображения (O(1)).
    """
    N = L * L
    
    # Сумма яркостей
    int_img = get_integral_image(channel_img)
    f_sum = get_window_sum(int_img, L)
    
    # Сумма квадратов яркостей (для дисперсии)
    int_sq_img = get_integral_image(channel_img**2)
    f_sq_sum = get_window_sum(int_sq_img, L)
    
    mean = f_sum / N
    # Формула дисперсии: E[X^2] - (E[X])^2
    var = (f_sq_sum / N) - (mean**2)
    
    return mean, np.maximum(var, 0)

def compute_indices(img_10ch):
    """
    Вычисляет спектральные индексы на основе 10 каналов MultiSenGE.
    Порядок: B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12
    Индексы: 0,  1,  2,  3,  4,  5,  6,  7,   8,   9
    """
    eps = 1e-8
    red = img_10ch[2]   # B4
    nir = img_10ch[3]   # B8
    swir = img_10ch[8]  # B11
    
    # NDVI - Индекс растительности
    ndvi = (nir - red) / (nir + red + eps)
    
    # NDBI - Индекс застройки
    ndbi = (swir - nir) / (swir + nir + eps)
    
    return ndvi, ndbi

def quantize_image(img, levels=16):
    """
    Квантование изображения (сжатие уровней яркости).
    Необходимо для корректного построения матриц GLCM.
    """
    img_min = np.min(img)
    img_max = np.max(img)
    # Нормализуем и приводим к целым числам от 0 до levels-1
    res = (img - img_min) / (img_max - img_min + 1e-8) * (levels - 1)
    return res.astype(np.uint8)

def compute_glcm_features(img, L, levels=16):
    """
    Расчет текстурных признаков GLCM в скользящем окне.
    Вычисляет Контраст и Однородность.
    """
    q_img = quantize_image(img, levels=levels)
    pad = L // 2
    h, w = q_img.shape
    
    contrast = np.zeros_like(img, dtype=np.float32)
    homogeneity = np.zeros_like(img, dtype=np.float32)
    
    # Проход скользящим окном
    # Для ускорения можно увеличить шаг (step) в range
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            window = q_img[i-pad:i+pad+1, j-pad:j+pad+1]
            
            # Матрица смежности (расстояние 1, угол 0 градусов)
            glcm = graycomatrix(window, distances=[1], angles=[0], 
                                levels=levels, symmetric=True, normed=True)
            
            # Извлекаем признаки
            contrast[i, j] = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity[i, j] = graycoprops(glcm, 'homogeneity')[0, 0]
            
    return contrast, homogeneity