import numpy as np
import time

def cascade_recursive_method(image, win):
    """
    Каскадная рекурсивная обработка O(1).
    Используется float64 для предотвращения переполнения при расчете сумм.
    """
    img_float = image.astype(np.float64)
    
    # Этап 1: Суммирование по строкам
    s_row = np.cumsum(img_float, axis=1)
    res_row = s_row.copy()
    res_row[:, win:] = s_row[:, win:] - s_row[:, :-win]
    
    # Этап 2: Суммирование результата по столбцам
    s_col = np.cumsum(res_row, axis=0)
    res_col = s_col.copy()
    res_col[win:, :] = s_col[win:, :] - s_col[:-win, :]
    
    return res_col

def fast_cascade_mean_std(image, window_size):
    """
    Расчет локального среднего и СКО (стандартного отклонения).
    """
    L = window_size
    area = L * L
    
    # Расчет среднего
    sum_img = cascade_recursive_method(image, L)
    mean_img = sum_img / area
    
    # Расчет среднего квадратов для дисперсии
    sum_sq_img = cascade_recursive_method(image**2, L)
    mean_sq_img = sum_sq_img / area
    
    # Дисперсия и СКО (защита от отрицательных чисел из-за точности float)
    var_img = np.maximum(mean_sq_img - mean_img**2, 0)
    std_img = np.sqrt(var_img)
    
    return mean_img, std_img

def smart_normalize(data, feature_type='mean', low_q=5, high_q=95):
    """
    Устойчивая нормализация признаков в формат uint8.
    'mean'   - прямое приведение (для яркостей).
    'std'    - растяжение по квантилям (отсечение выбросов).
    'signed' - сдвиг на 128 (для корреляций и индексов).
    """
    data_clean = np.nan_to_num(data)
    
    if feature_type == 'mean':
        # Для среднего нормализация не требуется, только приведение типов
        return np.clip(data_clean, 0, 255).astype(np.uint8)
    
    elif feature_type == 'std':
        # Устойчивое растяжение по квантилям (5% и 95%)
        low = np.percentile(data_clean, low_q)
        high = np.percentile(data_clean, high_q)
        
        if high <= low:
            return np.zeros_like(data_clean, dtype=np.uint8)
        
        normalized = (data_clean - low) / (high - low) * 255
        return np.clip(normalized, 0, 255).astype(np.uint8)
    
    elif feature_type == 'signed':
        # Сдвиг на 128 для знаковых данных (например, диапазон -1..1)
        # Масштабируем так, чтобы 0 стал 128
        return np.clip(data_clean * 127 + 128, 0, 255).astype(np.uint8)

def make_feature_sandwich(features_dict):
    """
    Сборка всех рассчитанных слоев в единый пакет признаков (датасет).
    features_dict: словарь {'Название': массив}
    """
    names = list(features_dict.keys())
    arrays = [features_dict[name] for name in names]
    sandwich = np.stack(arrays, axis=-1)
    return sandwich, names

def run_benchmark():
    """Тест производительности: Интегральное изображение vs Каскад."""
    size = 3000
    test_data = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    windows = [7, 15, 31, 63]
    
    print(f"Сравнение методов на изображении {size}x{size}:")
    print(f"{'Окно':<7} | {'Интегральный (сек)':<20} | {'Каскадный (сек)':<20} | {'Разница'}")
    print("-" * 75)
    
    for w in windows:
        # Интегральный метод
        t1 = time.time()
        integral = np.cumsum(np.cumsum(test_data.astype(np.float64), axis=0), axis=1)
        _ = np.zeros_like(test_data, dtype=np.float64)
        # (упрощенная логика для замера)
        dt1 = time.time() - t1
        
        # Каскадный метод
        t2 = time.time()
        _ = cascade_recursive_method(test_data, w)
        dt2 = time.time() - t2
        
        diff = ((dt1 - dt2) / dt1) * 100
        print(f"{w:<7} | {dt1:<20.5f} | {dt2:<20.5f} | {diff:>6.1f}% быстрее")

if __name__ == "__main__":
    # Запуск теста скорости
    run_benchmark()
    
    # Демонстрация сборки "сэндвича"
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    m_f64, s_f64 = fast_cascade_mean_std(img, 15)
    
    # Нормализация
    m_u8 = smart_normalize(m_f64, 'mean')
    s_u8 = smart_normalize(s_f64, 'std')
    
    # Сборка в пакет
    dataset, feature_names = make_feature_sandwich({
        'Original': img,
        'Local_Mean': m_u8,
        'Local_Std': s_u8
    })
    
    print(f"\nСформирован пакет признаков: {dataset.shape}")
    print(f"Состав признаков: {feature_names}")