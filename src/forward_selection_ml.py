import numpy as np
import os
import rasterio
from fast_features import extract_all_features, make_feature_sandwich
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def forward_selection_ml(dataset, mask, target_classes=None, eps=0.001, max_features=10):
    """
    Жадный отбор признаков по приросту точности kNN (5-fold CV) для многоклассовой задачи.
    """
    h, w, c = dataset.shape
    mask_flat = mask.flatten()
    feat_matrix = dataset.reshape(-1, c)
    
    # 1. Фильтруем валидные пиксели (фон = 0)
    valid_mask = mask_flat > 0
    X_all = feat_matrix[valid_mask]
    y_all = mask_flat[valid_mask]
    
    # 2. Оставляем только целевые классы (если указаны)
    if target_classes:
        target_mask = np.isin(y_all, target_classes)
        X_all = X_all[target_mask]
        y_all = y_all[target_mask]
        
    # 3. Ограничиваем выборку для ускорения CV (kNN чувствителен к N)
    MAX_SAMPLES = 20000
    if len(X_all) > MAX_SAMPLES:
        idx = np.random.choice(len(X_all), MAX_SAMPLES, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]
        
    print(f"📊 Данные: {len(X_all)} пикселей, {c} признаков, {len(np.unique(y_all))} классов")
    
    # 4. Масштабирование (критично для kNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    selected = []
    current_acc = 0.0
    history = []
    
    print(" Запуск ML-отбора (kNN k=5, 5-fold CV)...")
    
    for step in range(max_features):
        best_feat = -1
        best_acc = -1.0
        best_gain = -1.0
        
        for i in range(c):
            if i in selected: continue
            current_indices = selected + [i]
            X_subset = X_scaled[:, current_indices]
            
            # kNN классификатор
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            # micro average лучше для мультикласса
            scores = cross_val_score(clf, X_subset, y_all, cv=5, scoring='accuracy')
            acc = scores.mean()
            gain = acc - current_acc
            
            if gain > best_gain:
                best_gain = gain
                best_feat = i
                best_acc = acc
                
        if best_gain < eps or best_feat == -1:
            print(f"⏹️ Остановка на шаге {step}. Прирост точности {best_gain:.4f} < {eps}")
            break
            
        selected.append(best_feat)
        current_acc = best_acc
        history.append(current_acc)
        print(f"  Шаг {step+1}: добавлен признак #{best_feat}, Accuracy = {current_acc:.4f} (+{best_gain:.4f})")
        
    return selected, history

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    s2_dir = os.path.join(data_dir, "s2_pref")
    mask_dir = os.path.join(data_dir, "ground_reference")
    
    print(f"📂 Поиск данных в: {data_dir}")
    
    s2_files = [f for f in os.listdir(s2_dir) if f.endswith(('.tif', '.TIF'))]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.tif', '.TIF'))]
    
    if not s2_files or not mask_files:
        print("❌ Ошибка: Файлы не найдены!")
        return
    
    img_path = os.path.join(s2_dir, s2_files[0])
    mask_path = os.path.join(mask_dir, mask_files[0])
    print(f"Используемые файлы:\n   - {os.path.basename(img_path)}\n   - {os.path.basename(mask_path)}")
    
    # Загрузка
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
        print(f"❌ Ошибка при чтении файлов: {e}")
        return

    if mask.shape != img.shape:
        from scipy.ndimage import zoom
        mask = zoom(mask, (img.shape[0]/mask.shape[0], img.shape[1]/mask.shape[1]), order=0)

    # Генерация признаков
    print("\n⚙️ Генерация признаков (окно 15x15)...")
    feat_dict = extract_all_features(img, window_size=15)
    dataset, names = make_feature_sandwich(feat_dict)
    print(f"📦 Сэндвич: {dataset.shape}, Признаки: {names}")
    
    #  ДИНАМИЧЕСКИЙ ВЫБОР ВСЕХ КЛАССОВ ИЗ МАСКИ
    unique_classes = np.unique(mask)
    target_classes = [c for c in unique_classes if c > 0] # Исключаем фон (0)
    print(f"🎯 Анализ для классов: {target_classes}")
    
    # Запуск ML-отбора
    selected_indices, history = forward_selection_ml(dataset, mask, target_classes=target_classes)
    
    if selected_indices:
        selected_names = [names[i] for i in selected_indices]
        print(f"\n🏆 ИТОГОВЫЙ НАБОР (ML/kNN):")
        for i, name in enumerate(selected_names, 1):
            print(f"   {i}. {name}")
    else:
        print("\n️ Признаки не отобраны.")
    
    # Визуализация прироста точности
    if history:
        plt.figure(figsize=(9, 5))
        plt.plot(range(1, len(history)+1), history, marker='s', color='orange', linewidth=2, markersize=8)
        plt.title('Прирост точности классификации (Forward Selection + kNN)\nВсе классы маски', fontsize=12)
        plt.xlabel('Количество признаков', fontsize=11)
        plt.ylabel('Accuracy (5-fold CV)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, len(history)+1))
        plt.tight_layout()
        
        out_path = os.path.join(project_root, 'output', 'forward_selection_ml_all_classes.png')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        print(f"\n📊 График сохранён: {out_path}")
        plt.show()

if __name__ == "__main__":
    main()