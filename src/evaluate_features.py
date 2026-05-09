import numpy as np
import os
import rasterio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.spatial import distance
# Импортируем обновленную функцию из твоего модуля
from fast_features import extract_all_features 

def calculate_class_stats(dataset, mask, class_id):
    """Извлекает пиксели определенного класса и считает их статистику."""
    mask_flat = mask.flatten()
    # dataset имеет форму (H, W, C), превращаем в (N, C)
    dataset_reshaped = dataset.reshape(-1, dataset.shape[-1])
    
    pixels = dataset_reshaped[mask_flat == class_id]
    
    if len(pixels) < 5:
        print(f"   Класс {class_id}: недостаточно пикселей ({len(pixels)})")
        return None, None
    
    print(f"   Класс {class_id}: {len(pixels)} пикселей, признаков: {pixels.shape[1]}")
    
    mean_vec = np.mean(pixels, axis=0)
    cov_mat = np.cov(pixels, rowvar=False)
    
    # Регуляризация ковариационной матрицы для стабильности
    cov_mat += np.eye(cov_mat.shape[0]) * 1e-6
    
    return mean_vec, cov_mat

def mahalanobis_distance(m1, m2, cov1, cov2):
    """Вычисляет расстояние Махаланобиса при допущении общей ковариации."""
    avg_cov = (cov1 + cov2) / 2
    try:
        inv_cov = np.linalg.inv(avg_cov)
        diff = m1 - m2
        dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        return dist
    except np.linalg.LinAlgError:
        return np.nan

def bhattacharyya_distance(m1, m2, cov1, cov2):
    """
    Вычисляет расстояние Бхаттачария. 
    Учитывает различие в ковариационных матрицах (форме эллипсоидов).
    """
    cov = (cov1 + cov2) / 2
    diff = m1 - m2
    try:
        # Первая часть: разность средних (похожа на Махаланобиса)
        term1 = (1/8) * np.dot(np.dot(diff, np.linalg.inv(cov)), diff)
        # Вторая часть: разность объемов и ориентации эллипсоидов
        term2 = (1/2) * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
        return term1 + term2
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

def main():
    # 1. АВТОПОИСК ПУТЕЙ
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    
    print(f"--- Сканирую папку: {data_dir} ---")
    all_files = glob.glob(os.path.join(data_dir, "**", "*.*"), recursive=True)
    
    mask_path = None
    img_path = None

    for f in all_files:
        fname = os.path.basename(f).lower()
        if "31ufp_gr" in fname and fname.endswith('.tif') and "_0_0.tif" in fname:
            if mask_path is None: mask_path = f
        if "31ufp" in fname and "_gr_" not in fname and fname.endswith('.tif'):
            if img_path is None: img_path = f

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

    # 2. ЗАГРУЗКА И ПОДГОТОВКА
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        with rasterio.open(img_path) as src:
            image_data = src.read()
            # Берем среднее по каналам для текстурного анализа (как в ревизии)
            image = np.mean(image_data, axis=0).astype(np.float32)
            
            # Нормализация 0-255
            img_min, img_max = image.min(), image.max()
            image = ((image - img_min) / (img_max - img_min)) * 255
            image = image.astype(np.uint8)
    except Exception as e:
        print(f"Ошибка при чтении: {e}")
        return

    # Масштабирование маски если нужно
    if mask.shape != image.shape:
        from scipy.ndimage import zoom
        mask = zoom(mask, (image.shape[0]/mask.shape[0], image.shape[1]/mask.shape[1]), order=0)

    # 3. ГЕНЕРАЦИЯ РАСШИРЕННЫХ ПРИЗНАКОВ (РЕВИЗИЯ 3.0)
    print(f"\nГенерация признаков (Std, Rho_Avg, Rho_Range)...")
    feature_dict = extract_all_features(image, window_size=15)
    
    # Собираем все признаки в один массив (H, W, C)
    feat_names = list(feature_dict.keys())
    feat_channels = [feature_dict[name] for name in feat_names]
    dataset = np.stack(feat_channels, axis=-1)
    
    print(f"   Сформировано признаков: {feat_names}")

    # 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
    classes = np.unique(mask)
    classes = classes[classes > 0]
    stats = {}
    
    for cls in classes:
        print(f"\nОбработка класса {cls}:")
        mean_vec, cov_mat = calculate_class_stats(dataset, mask, cls)
        if mean_vec is not None:
            stats[cls] = {'mean': mean_vec, 'cov': cov_mat}

    # 5. СРАВНЕНИЕ КЛАССОВ
    print(f"\n{'='*60}\nРЕЗУЛЬТАТЫ АНАЛИЗА\n{'='*60}")
    
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]
            if c1 not in stats or c2 not in stats: continue
            
            m1, cov1 = stats[c1]['mean'], stats[c1]['cov']
            m2, cov2 = stats[c2]['mean'], stats[c2]['cov']
            
            # Махаланобис по всему вектору
            dist_m = mahalanobis_distance(m1, m2, cov1, cov2)
            # Бхаттачария по всему вектору
            dist_b = bhattacharyya_distance(m1, m2, cov1, cov2)
            
            print(f"\nПара {c1} и {c2}:")
            print(f"   Расстояние Махаланобиса: {dist_m:.4f}")
            print(f"   Расстояние Бхаттачария:  {dist_b:.4f}")
            
            # Сравнение информативности (Яркость vs Текстура)
            # Оригинал (индекс 0) против Rho_Avg (допустим индекс 3)
            print(f"   Средние (Original): {m1[0]:.2f} vs {m2[0]:.2f}")

    # 6. ВИЗУАЛИЗАЦИЯ
    if 2 in stats and 11 in stats:
        # Подготовка данных для графиков
        p2 = dataset[mask == 2]
        p11 = dataset[mask == 11]
        df_plot = pd.DataFrame(np.vstack([p2, p11]), columns=feat_names)
        df_plot['class'] = [2]*len(p2) + [11]*len(p11)

        # График 1: Гистограммы Rho_Avg
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.histplot(data=df_plot, x='Rho_Avg', hue='class', kde=True, element="step", ax=ax1)
        ax1.set_title('Распределение текстурной корреляции (Rho_Avg)')

        # График 2: Совместное распределение (Эллипсоиды)
        # Смотрим Яркость vs Std или Rho_Avg
        sns.kdeplot(data=df_plot, x='Original', y='Rho_Avg', hue='class', fill=True, alpha=0.3, ax=ax2)
        ax2.set_title('Совместное распределение (Эллипсоиды рассеяния)\nOriginal vs Rho_Avg')

        plt.tight_layout()
        plt.savefig('analysis_complex.png')
        print("\n[УСПЕХ] Комплексный анализ сохранен в analysis_complex.png")
        plt.show()

if __name__ == "__main__":
    main()