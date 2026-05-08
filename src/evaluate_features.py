import numpy as np
import os
import rasterio
from scipy.spatial import distance
from fast_features import fast_cascade_mean_std, smart_normalize, make_feature_sandwich

def calculate_class_stats(dataset, mask, class_id):
    """Извлекает пиксели определенного класса и считает их статистику."""
    mask_flat = mask.flatten()
    dataset_reshaped = dataset.reshape(-1, dataset.shape[-1])
    
    pixels = dataset_reshaped[mask_flat == class_id]
    
    if len(pixels) < 3:
        print(f"  Класс {class_id}: недостаточно пикселей ({len(pixels)})")
        return None, None
    
    print(f"  Класс {class_id}: {len(pixels)} пикселей, признаков: {pixels.shape[1]}")
    
    mean_vec = np.mean(pixels, axis=0)
    cov_mat = np.cov(pixels, rowvar=False)
    
    # Регуляризация для устойчивости
    cov_mat += np.eye(cov_mat.shape[0]) * 1e-6
    
    return mean_vec, cov_mat

def mahalanobis_distance(m1, m2, cov1, cov2):
    """Вычисляет расстояние Махаланобиса между двумя распределениями."""
    avg_cov = (cov1 + cov2) / 2
    avg_cov += np.eye(avg_cov.shape[0]) * 1e-6
    
    try:
        inv_cov = np.linalg.inv(avg_cov)
        diff = m1 - m2
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        return dist
    except np.linalg.LinAlgError:
        return np.linalg.norm(diff)

def main():
    import matplotlib.pyplot as plt
    import glob

    # 1. АВТОПОИСК ПУТЕЙ
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print(f"--- Сканирую папку: {data_dir} ---")
    
    all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    
    mask_path = None
    img_path = None

    for f in all_files:
        fname = os.path.basename(f).lower()
        
        # Маска - файл, где ЕСТЬ '_gr_' и заканчивается на _0_0.tif
        if "31ufp_gr" in fname and fname.endswith('.tif') and "_0_0.tif" in fname:
            if mask_path is None:
                mask_path = f
        
        # ИЗОБРАЖЕНИЕ - файл, где НЕТ '_gr_', но есть '31ufp' (это Sentinel-2 данные)
        if "31ufp" in fname and "_gr_" not in fname and fname.endswith('.tif'):
            if img_path is None:
                img_path = f

    # Если не нашли Sentinel, ищем любой другой .tif кроме маски
    if not img_path:
        for f in all_files:
            fname = os.path.basename(f).lower()
            if fname.endswith('.tif') and f != mask_path and "_gr_" not in fname:
                img_path = f
                break

    if not mask_path or not img_path:
        print("Ошибка: не найдены нужные файлы!")
        return

    print(f"Использую маску: {os.path.basename(mask_path)}")
    print(f"Использую изображение: {os.path.basename(img_path)}")

    # 2. ЗАГРУЗКА
    try:
        # Загружаем маску
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            print(f"Маска: форма {mask.shape}, dtype {mask.dtype}")
            print(f"  Уникальные значения: {np.unique(mask)}")
        
        # Загружаем изображение Sentinel-2
        with rasterio.open(img_path) as src:
            # Читаем все каналы
            image_data = src.read()
            print(f"Изображение: форма {image_data.shape}, dtype {image_data.dtype}")
            print(f"  Количество каналов: {image_data.shape[0]}")
            
            # Sentinel-2: обычно каналы 2,3,4 - это RGB
            if image_data.shape[0] >= 3:
                # Берем RGB каналы (индексы 0,1,2 или 1,2,3 в зависимости от данных)
                # Для визуализации создаем RGB изображение
                rgb = np.stack([
                    image_data[0],  # красный канал
                    image_data[1],  # зеленый канал  
                    image_data[2]   # синий канал
                ], axis=-1)
                
                # Конвертируем в оттенки серого
                image = np.mean(rgb, axis=-1).astype(np.float32)
                print(f"  Создано RGB (3 канала) -> оттенки серого")
            else:
                # Если только один канал
                image = image_data[0].astype(np.float32)
                print(f"  Использую один канал")
            
            # Нормализация изображения в диапазон 0-255
            img_min, img_max = image.min(), image.max()
            print(f"  Диапазон до нормализации: min={img_min:.2f}, max={img_max:.2f}")
            
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min)) * 255
                image = image.astype(np.uint8)
                print(f"  Диапазон после нормализации: {image.min()} - {image.max()}")
            else:
                print(f"  ОШИБКА: Все значения изображения одинаковы!")
                # Создаем тестовое изображение
                image = np.random.randint(0, 255, mask.shape, dtype=np.uint8)
                print(f"  Создано тестовое изображение")
        
    except Exception as e:
        print(f"Ошибка при чтении: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. ПРОВЕРКА РАЗМЕРОВ
    print(f"\nРазмеры: маска {mask.shape}, изображение {image.shape}")
    
    if mask.shape != image.shape:
        print(f"Внимание: размеры не совпадают!")
        from scipy.ndimage import zoom
        factors = (image.shape[0] / mask.shape[0], image.shape[1] / mask.shape[1])
        print(f"  Масштабируем маску с коэффициентом {factors}")
        
        from scipy.ndimage import map_coordinates
        h, w = mask.shape
        y_coords = np.linspace(0, h-1, image.shape[0])
        x_coords = np.linspace(0, w-1, image.shape[1])
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.array([yy.ravel(), xx.ravel()])
        mask_resampled = map_coordinates(mask.astype(float), coords, order=0).reshape(image.shape)
        mask = np.round(mask_resampled).astype(mask.dtype)
        print(f"  После масштабирования: маска {mask.shape}")

    # 4. ПРОВЕРКА ДАННЫХ
    print(f"\nПроверка данных:")
    for cls in np.unique(mask):
        if cls > 0:
            pixels_in_class = np.sum(mask == cls)
            if pixels_in_class > 0:
                img_values = image[mask == cls]
                print(f"  Класс {cls}: {pixels_in_class} пикселей, средняя яркость = {np.mean(img_values):.2f}")

    # 5. ГЕНЕРАЦИЯ ПРИЗНАКОВ
    print(f"\nГенерация текстурных признаков...")
    mean_f, std_f = fast_cascade_mean_std(image.astype(np.float32), 15)
    mean_u8 = smart_normalize(mean_f, 'mean')
    std_u8 = smart_normalize(std_f, 'std')
    
    dataset, names = make_feature_sandwich({
        'Original': image,
        'Mean': mean_u8,
        'Std': std_u8
    })
    
    print(f"  Формат признаков: {dataset.shape}")
    print(f"  Диапазон значений: min={dataset.min():.2f}, max={dataset.max():.2f}")

    # 6. СТАТИСТИЧЕСКИЙ АНАЛИЗ
    classes = np.unique(mask)
    classes = classes[classes > 0]
    
    print(f"\nНайдено классов: {classes}")
    
    if len(classes) < 2:
        print("Ошибка: нужно минимум 2 класса")
        return
    
    # Собираем статистику
    stats = {}
    for cls in classes:
        print(f"\nОбработка класса {cls}:")
        mean_vec, cov_mat = calculate_class_stats(dataset, mask, cls)
        if mean_vec is not None:
            stats[cls] = {'mean': mean_vec, 'cov': cov_mat}
            print(f"  Среднее: [{mean_vec[0]:.2f}, {mean_vec[1]:.2f}, {mean_vec[2]:.2f}]")
    
    # Анализ пар
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print('='*60)
    
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]
            
            if c1 not in stats or c2 not in stats:
                print(f"\nПара {c1}-{c2}: недостаточно данных")
                continue
            
            m1 = stats[c1]['mean']
            m2 = stats[c2]['mean']
            cov1 = stats[c1]['cov']
            cov2 = stats[c2]['cov']
            
            # Евклидово расстояние
            euclidean_dist = np.linalg.norm(m1 - m2)
            
            # Расстояние только по яркости
            dist_orig = mahalanobis_distance(m1[0:1], m2[0:1], 
                                            cov1[0:1, 0:1], 
                                            cov2[0:1, 0:1])
            
            # Полное расстояние
            dist_full = mahalanobis_distance(m1, m2, cov1, cov2)
            
            print(f"\nПара {c1} и {c2}:")
            print(f"  Евклидово расстояние: {euclidean_dist:.4f}")
            print(f"  Яркость: {dist_orig:.4f}")
            print(f"  Полный вектор: {dist_full:.4f}")
            
            if euclidean_dist > 0:
                print(f"  Средние значения:")
                print(f"    Класс {c1}: [{m1[0]:.2f}, {m1[1]:.2f}, {m1[2]:.2f}]")
                print(f"    Класс {c2}: [{m2[0]:.2f}, {m2[1]:.2f}, {m2[2]:.2f}]")
    
    # 7. ВИЗУАЛИЗАЦИЯ (Гистограммы распределения)
    if 2 in stats and 11 in stats:
        import matplotlib.pyplot as plt
        
        # Извлекаем признаки для классов 2 и 11
        p2 = dataset[mask == 2]
        p11 = dataset[mask == 11]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # График 1: Исходная яркость
        ax1.hist(p2[:, 0], bins=50, alpha=0.5, label='Класс 2 (Поля)', color='blue', density=True)
        ax1.hist(p11[:, 0], bins=50, alpha=0.5, label='Класс 11 (Застройка)', color='red', density=True)
        ax1.set_title('Распределение яркости (Original)\nКлассы сильно пересекаются')
        ax1.set_xlabel('Значение яркости')
        ax1.legend()
        
        # График 2: Текстурный признак (Std)
        ax2.hist(p2[:, 2], bins=50, alpha=0.5, label='Класс 2 (Поля)', color='blue', density=True)
        ax2.hist(p11[:, 2], bins=50, alpha=0.5, label='Класс 11 (Застройка)', color='red', density=True)
        ax2.set_title('Распределение текстуры (Std)\nКлассы разделились!')
        ax2.set_xlabel('Значение СКО (Std)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('analysis_results.png')
        print("\n[УСПЕХ] Графики сохранены в файл analysis_results.png")
        plt.show()

if __name__ == "__main__":
    main()