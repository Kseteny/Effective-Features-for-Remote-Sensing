import numpy as np
import time

def cascade_recursive_method(image, win):
    """
    Каскадная рекурсивная обработка O(1).
    Используем float64 для предотвращения переполнения при суммировании.
    """
    img_float = image.astype(np.float64)
    
    # Этап 1: Суммирование по строкам
    s_row = np.cumsum(img_float, axis=1)
    res_row = s_row.copy()
    res_row[:, win:] = s_row[:, win:] - s_row[:, :-win]
    
    # Этап 2: Суммирование по столбцам
    s_col = np.cumsum(res_row, axis=0)
    res_col = s_col.copy()
    res_col[win:, :] = s_col[win:, :] - s_col[:-win, :]
    
    return res_col

def fast_cascade_mean_std(image, window_size):
    """Расчет локального среднего и СКО через каскадную схему."""
    L = window_size
    area = L * L
    
    sum_img = cascade_recursive_method(image, L)
    mean_img = sum_img / area
    
    sum_sq_img = cascade_recursive_method(image**2, L)
    mean_sq_img = sum_sq_img / area
    
    # Дисперсия: E[X^2] - (E[X])^2
    var_img = np.maximum(mean_sq_img - mean_img**2, 0)
    std_img = np.sqrt(var_img)
    
    return mean_img, std_img

def smart_normalize(data, feature_type='mean'):
    """
    Адаптивная нормализация (uint8) по рекомендациям В.В. Сергеева.
    """
    data_clean = np.nan_to_num(data)
    
    if feature_type == 'mean':
        # Для среднего нормализация избыточна, если вход в 0-255.
        # Просто приводим к байту с обрезкой хвостов float.
        return np.clip(data_clean, 0, 255).astype(np.uint8)
    
    elif feature_type == 'std':
        # Для СКО используем растяжение (stretch), так как диапазон специфичен.
        d_min, d_max = np.min(data_clean), np.max(data_clean)
        if d_max <= d_min: return np.zeros_like(data_clean, dtype=np.uint8)
        return ((data_clean - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    
    elif feature_type == 'signed':
        # Сдвиг на 128 для знаковых признаков (корреляции, разности).
        # Центрируем 0 в 128.
        return np.clip(data_clean * 127 + 128, 0, 255).astype(np.uint8)

def run_benchmark():
    """Сравнение производительности: Интегральный метод vs Каскад."""
    # Тестовое изображение 3000x3000
    test_data = np.random.randint(0, 256, (3000, 3000), dtype=np.uint8)
    windows = [3, 7, 15, 31, 63]
    
    print(f"\nСравнительный тест: Интегральный метод vs Каскадная рекурсия")
    print(f"{'Окно':<7} | {'Интегральный (сек)':<20} | {'Каскадный (сек)':<20} | {'Разница'}")
    print("-" * 75)
    
    for w in windows:
        # Интегральный метод (классика)
        t1 = time.time()
        integral = np.cumsum(np.cumsum(test_data.astype(np.float64), axis=0), axis=1)
        res_int = np.zeros_like(test_data, dtype=np.float64)
        res_int[w:, w:] = (integral[w:, w:] + integral[:-w, :-w] - 
                           integral[:-w, w:] - integral[w:, :-w])
        dt1 = time.time() - t1
        
        # Каскадный метод
        t2 = time.time()
        _ = cascade_recursive_method(test_data, w)
        dt2 = time.time() - t2
        
        diff = ((dt1 - dt2) / dt1) * 100
        print(f"{w:<7} | {dt1:<20.5f} | {dt2:<20.5f} | {diff:>6.1f}% быстрее")

if __name__ == "__main__":
    run_benchmark()
    
    # Пример использования нормализации
    test_img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    m_f64, s_f64 = fast_cascade_mean_std(test_img, 7)
    
    m_u8 = smart_normalize(m_f64, 'mean')
    s_u8 = smart_normalize(s_f64, 'std')
    
    print(f"\nПроверка нормализации:")
    print(f"Mean float64 max: {m_f64.max():.2f} -> uint8 max: {m_u8.max()}")
    print(f"Std float64 max: {s_f64.max():.2f} -> uint8 max: {s_u8.max()}")