import numpy as np
from scipy.ndimage import uniform_filter

def get_fast_stats(image, window_size):
    """
    Вычисляет локальное среднее и СКО с вычислительной сложностью O(1).
    Реализует пункты 2.1 и 2.2 Ревизии[cite: 30, 36, 37].
    """
    n = window_size
    # Fsum и Fsq_sum 
    mean = uniform_filter(image, size=n, mode='mirror')
    sq_mean = uniform_filter(image**2, size=n, mode='mirror')
    
    # Формула дисперсии: σ² = Fsq_sum/N - μ² [cite: 37]
    var = sq_mean - mean**2
    var = np.maximum(var, 0)  # Защита от отрицательных значений из-за точности float
    std = np.sqrt(var)
    
    return mean, var, std

def calc_directional_rho(image, mean, var, window_size):
    """
    Вычисляет коэффициенты корреляции по 4 направлениям (0°, 90°, 45°, 135°).
    Реализует формулы из п. 2.1 и 2.2 Ревизии[cite: 30, 40].
    """
    n = window_size
    N = n**2
    # Для стабильности добавляем малую константу в знаменатель
    eps = 1e-10 
    
    # Направления смещения (dy, dx)
    directions = {
        '0': (0, 1),    # Горизонтально
        '90': (1, 0),   # Вертикально
        '45': (1, 1),   # Диагональ 1
        '135': (1, -1)  # Диагональ 2
    }
    
    rhos = {}
    
    for angle, (dy, dx) in directions.items():
        # Сдвигаем изображение для вычисления Fadj (смежных произведений) 
        # Используем np.roll для быстрого смещения
        shifted = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
        
        # Вычисляем локальную сумму смежных произведений Fadj 
        f_adj = uniform_filter(image * shifted, size=n, mode='mirror')
        
        # Вычисляем rho_dir = (1/N * Fadj - mu²) / sigma² [cite: 40]
        rho = (f_adj - mean**2) / (var + eps)
        rhos[angle] = np.clip(rho, -1, 1) # Корреляция всегда в пределах [-1, 1]
        
    return rhos

def extract_all_features(image, window_size=15):
    """
    Основная функция для генерации избыточного признакового пространства[cite: 2].
    """
    # 1. Базовые статистики [cite: 26]
    mean, var, std = get_fast_stats(image, window_size)
    
    # 2. Направленные корреляции [cite: 39, 40]
    rhos = calc_directional_rho(image, mean, var, window_size)
    
    # 3. Производные признаки [cite: 34]
    # Средняя корреляция (общая упорядоченность) [cite: 41, 43]
    rho_avg = (rhos['0'] + rhos['90'] + rhos['45'] + rhos['135']) / 4
    
    # Размах корреляции (анизотропия/направленность) [cite: 42, 44]
    rho_max = np.maximum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']])
    rho_min = np.minimum.reduce([rhos['0'], rhos['90'], rhos['45'], rhos['135']])
    rho_range = rho_max - rho_min
    
    # Формируем итоговый словарь признаков
    feature_space = {
        'Original': image,
        'Mean': mean,
        'Std': std,
        'Rho_Avg': rho_avg,
        'Rho_Range': rho_range,
        'Rho_0': rhos['0'],
        'Rho_90': rhos['90']
    }
    
    return feature_space

if __name__ == "__main__":
    print("Модуль fast_features готов к работе согласно Ревизии 3.0[cite: 1].")