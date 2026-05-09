import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from fast_features import extract_all_features, make_feature_sandwich
from forward_selection_stats import forward_selection_bhatta
from forward_selection_ml import forward_selection_ml
import rasterio
from scipy.ndimage import zoom

def main():
    print("="*60)
    print("🔬 СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ОТБОРА ПРИЗНАКОВ")
    print("="*60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    s2_dir = os.path.join(data_dir, "s2_pref")
    mask_dir = os.path.join(data_dir, "ground_reference")
    
    # Загрузка данных (берём первый файл для демонстрации)
    s2_files = [f for f in os.listdir(s2_dir) if f.endswith(('.tif', '.TIF'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.TIF'))]
    
    img_path = os.path.join(s2_dir, s2_files[0])
    mask_path = os.path.join(mask_dir, mask_files[0])
    
    print(f"📂 Используемые данные:\n   - {os.path.basename(img_path)}\n   - {os.path.basename(mask_path)}")
    
    # Загрузка изображений
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        with rasterio.open(img_path) as src:
            img_data = src.read()
            img = np.mean(img_data, axis=0).astype(np.float32) if img_data.ndim == 3 else img_data.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min)) * 255
            img = img.astype(np.uint8)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    if mask.shape != img.shape:
        mask = zoom(mask, (img.shape[0]/mask.shape[0], img.shape[1]/mask.shape[1]), order=0)

    # Генерация признаков
    print("\n⚙️ Генерация признакового пространства...")
    feat_dict = extract_all_features(img, window_size=15)
    dataset, names = make_feature_sandwich(feat_dict)
    
    unique_classes = [c for c in np.unique(mask) if c > 0]
    print(f"🎯 Анализ для классов: {unique_classes}")
    
    # --- ЗАПУСК МЕТОДА 1: СТАТИСТИЧЕСКИЙ (Бхаттачария) ---
    print("\n--- 📊 Метод 1: Статистический (Расстояние Бхаттачария) ---")
    selected_stats, _ = forward_selection_bhatta(dataset, mask, target_classes=(2, 11)) # Для честности берем ту же пару, что и раньше, или все классы
    
    # --- ЗАПУСК МЕТОДА 2: ML-ПОДХОД (kNN + CV) ---
    print("\n--- 🤖 Метод 2: ML-подход (Точность kNN, 5-fold CV) ---")
    selected_ml, _ = forward_selection_ml(dataset, mask, target_classes=unique_classes)
    
    # --- ФОРМИРОВАНИЕ РЕЗУЛЬТАТОВ ---
    stats_names = [names[i] for i in selected_stats]
    ml_names = [names[i] for i in selected_ml]
    
    # Создаём DataFrame для красивого вывода
    max_len = max(len(stats_names), len(ml_names))
    df = pd.DataFrame({
        'Шаг': range(1, max_len + 1),
        'Статистический (Бхаттачария)': stats_names + [''] * (max_len - len(stats_names)),
        'ML-подход (kNN Accuracy)': ml_names + [''] * (max_len - len(ml_names))
    })
    
    print("\n" + "="*60)
    print("🏆 ИТОГОВАЯ ТАБЛИЦА ОТОБРАННЫХ ПРИЗНАКОВ")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Анализ совпадений
    common = set(stats_names) & set(ml_names)
    only_stats = set(stats_names) - set(ml_names)
    only_ml = set(ml_names) - set(stats_names)
    
    print(f"\n📈 АНАЛИЗ СОВПАДЕНИЙ:")
    print(f"   ✅ Общие признаки ({len(common)}): {', '.join(common) if common else 'Нет'}")
    print(f"   📊 Только в статистике ({len(only_stats)}): {', '.join(only_stats) if only_stats else 'Нет'}")
    print(f"   🤖 Только в ML ({len(only_ml)}): {', '.join(only_ml) if only_ml else 'Нет'}")
    
    # Визуализация сравнения
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Раскраска заголовков
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50' if i == 1 else '#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
        
    plt.title('Сравнение методов отбора признаков\n(Forward Selection)', fontsize=14, pad=20)
    
    out_path = os.path.join(project_root, 'output', 'comparison_table.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📸 Таблица сохранена: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()